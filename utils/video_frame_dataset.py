# Copyright (c) Akash Chaurasia
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import itertools
import logging
from pathlib import Path
import random
from timeit import default_timer
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchvision.transforms import Compose, InterpolationMode, Normalize, RandomResizedCrop

from utils.transform import create_random_augment
from utils.retry import DataloadFailure, retry_random_idx_on_err


class InsufficientVideoLengthError(Exception):
    pass


def collate_batch(list_of_examples):
    xs = torch.stack([l[0] for l in list_of_examples if not isinstance(l, DataloadFailure)])
    ys = torch.stack([l[1] for l in list_of_examples if not isinstance(l, DataloadFailure)])

    return xs, ys

class VideoFrameDataset(Dataset):
    """
    Uniformly sample across videos, and within each video randomly sample a sequence of frames
    according to the given number of frames and stride.
    Args:
        root_path: The root path in which video folders lie.
        split: dataset split from the overall dataset (must be 'train', 'test', or 'val')
        num_frames: number of frames from a video constituting one example
        stride: number of frames between sampled frames. For example a stride of 4 starting at 0
            means we would take indices 0, 4, 8, etc.
        transform: WIP
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders.
        index_filename: name of index file in each video's folder. Something like 'index.pkl'
        is_eval: If True, we drop augmentations (maybe?)
    """

    # Default parameters for preprocessing (augmentation, crop, etc.)
    RAW_SIZE = (224, 224)
    AUTOAUGMENT_TYPE = 'rand-m1-n1-mstd0.5-inc0'
    INPUT_SIZE = 224
    CROP_SCALE = (0.55, 1)
    MEAN = (0.45, 0.45, 0.45)
    STD = (0.225, 0.225, 0.225)

    def __init__(
        self,
        ledger_path: str,
        num_frames: int = 8,
        stride: int = 4,
        do_augmentation=True,
        is_eval: bool = False,
    ):
        super().__init__()

        self.ledger_path = Path(ledger_path)
        self.num_frames = num_frames
        self.stride = stride
        self.is_eval = is_eval

        # Don't do augmentation for testing splits
        self.do_augmentation = do_augmentation and not self.is_eval
        self.rand_augment = create_random_augment(
            input_size=self.RAW_SIZE,
            auto_augment=self.AUTOAUGMENT_TYPE,
            interpolation='bicubic',
        )

        xforms = []
        # Not a huge deal probably but for sanity don't to RRC on test examples
        if not self.is_eval:
            xforms.append(
                RandomResizedCrop(
                    self.INPUT_SIZE,
                    scale=self.CROP_SCALE,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )

        self.transforms = Compose(xforms + [Normalize(self.MEAN, self.STD)])

        self.video_metadata = pd.read_csv(self.ledger_path)
        logging.info(f"Instantiated {self.__class__.__name__} based on {str(self.ledger_path)}")
        logging.info(f"Number of examples: {len(self)}")

    def _get_start_index(self, num_frames) -> int:
        if num_frames <= (self.num_frames - 1) * self.stride + 1:
            return 0
        return random.randint(0, num_frames - ((self.num_frames - 1) * self.stride))

    @retry_random_idx_on_err(do_retry=True)
    def __getitem__(self, idx):
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.
        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        # Get the number of frames in sampled video
        meta_row = self.video_metadata.iloc[idx]
        video_path = Path(meta_row['avi_path'])

        # Make inputs (C, T, H, W) for Conv3d
        return self._get_frames(video_path).permute(1, 0, 2, 3)

    def _get_frames(self, video_path: Union[Path, str], start_index: Optional[int] = None):
        """
        Loads the frames of a video at the corresponding
        indices.
        Args:
            video_path: Path to video for example
            start_index: index to start sampling from
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        reader = VideoReader(str(video_path))
        video_meta = reader.get_metadata()['video']

        num_frames = int(video_meta['fps'][0] * video_meta['duration'][0])

        # if num_frames < self.stride * (self.num_frames - 1) + 1:
        #     raise InsufficientVideoLengthError(
        #         f"Video {str(video_path)} has {num_frames} frames, which is "
        #         f"insufficient for parameters {self.num_frames=}, {self.stride=}"
        #     )

        start_index = self._get_start_index(num_frames)
        start_s = start_index / video_meta['fps'][0]
        reader.seek(start_s - 1e-5, keyframes_only=True)

        frames = []
        for frame_data in itertools.islice(
            reader,
            0,
            self.stride * (self.num_frames - 1) + 1,
            self.stride,
        ):
            frames.append(frame_data['data'])

        num_missing_frames = self.num_frames - len(frames)
        for _ in range(num_missing_frames):
            frames.append(torch.zeros((3, 224, 224)))

        frames = torch.stack(frames).float() / 255.0
        frames = self.transforms(frames)  # (T, C, H, W)

        return frames

    def __len__(self):
        return len(self.video_metadata)

    def get_debug(self, index=None):
        """ This is completely broken as of now """
        index = index if index is not None else random.randint(0, len(self))
        video = self.video_metadata[index]

        start_index = self._get_start_index(video)

        frames = [
            video.frames[idx].load()
            for idx in range(
                start_index, start_index + self.stride * (self.num_frames - 1) + 1, self.stride
            )
        ]

        augmented_frames = self.rand_augment(frames)

        raw_frames = torch.stack([self.to_tensor_transform(f) for f in frames]).float() / 255.0
        augmented_frames = torch.stack([
            self.to_tensor_transform(f) for f in augmented_frames
        ]).float() / 255.0

        preprocessed_frames = self.transforms(raw_frames)  # (T, C, H, W)
        preprocessed_augmented_frames = self.transforms(augmented_frames)

        return raw_frames, preprocessed_frames, augmented_frames, preprocessed_augmented_frames


if __name__ == '__main__':
    dataset = VideoFrameDataset('/scratch/users/akashc/test_dataset', split='train')

    start = default_timer()
    for i in range(100):
        frames, lbl = dataset[i]
        print(frames.shape, lbl)
    end = default_timer()

    print(f"Time for 100 iterations: {end - start:.3f} ({100 / (end - start):.3f} fps)")
