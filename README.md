# Masked Autoencoders As Spatiotemporal Learners
This is an unofficial PyTorch/GPU implementation of [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113).

```
@Article{STMaskedAutoencoders2022,
  author  = {Feichtenhofer, Christoph and Fan, Haoqi and Li, Yanghao and He, Kaiming},
  journal = {arXiv:2205.09113},
  title   = {Masked Autoencoders As Spatiotemporal Learners},
  year    = {2022},
}
```

## TODOs
- [x] Add pretraining code
- [x] Switch to learned spatiotemporal embedding
- [ ] Test code on surgical dataset (only 200 videos left to process)
- [ ] Add visualization code
- [ ] (OPTIONAL) Add prospective and retrospective frame masking 


## Getting Started
This repository runs on PyTorch 11.1 and above. To get started, clone the repository and install the required dependencies:
```
$ git clone https://github.com/cyrilzakka/MAE3D
$ cd MAE3D
$ pip install -r requirements.txt
```
Optionally, install `wandb` for training visualization:
```
$ pip install wandb
```

## Pretraining
### Dataset Preparation
In order to perform large-scale pretraining, your data should be organized in the following way:
```
dataset
│
├───ledger.csv
└───train 
     ├───video_1
     │     ├───img_00001.jpg
     │     .
     │     └───img_03117.jpg
     ├───video_2
     │     ├───img_00001.jpg
     │     .
     │     └───img_02744.jpg
     └───video_3
           ├───img_00001.jpg
           .
           └───img_0323.jpg
```
with the accompanying `ledger.csv` containing rows listing the `video_name`, `start_frame`, `end_frame` and `class/pseudoclass`:
```
video_1 1 3117 1
video_2 1 2744 0
video_3 1 323 0
```

### Dataloader
Fast and efficient loading of video data for training is done using the [VideoFrameDataset](https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch) library:

```python
dataset_train = VideoFrameDataset(root_path:str, annotationfile_path: str, num_segments:int, frames_per_segment:int, transform:None, test_mode:bool)
```
where each video is split into even `num_segments`, from which a random start index is sampled and `frames_per_segment` consecutive frames are loaded.

### Training
To train with the default `--model vit_large_patch16` for `--epochs 400` and a `--batch_size 8` at an `--input_size 224` run:
```
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py
```
More training options and parameters can be viewed and modified in `main_pretrain.py`.

### Visualization
Coming soon.

## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/cyrilzakka/mae3d/blob/main/LICENSE) for details.