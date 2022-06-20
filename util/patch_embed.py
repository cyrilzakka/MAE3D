# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Referenced from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from torch import nn as nn
from torch import _assert

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                num_frames=16, temp_stride=2, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (num_frames, img_size, img_size)
        patch_size = (temp_stride, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.temp_stride = temp_stride
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, T, H, W = x.shape
        _assert(H == self.img_size[1], f"Input height ({H}) doesn't match model ({self.img_size[1]}).")
        _assert(W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]}).")
        _assert(T == self.img_size[0], f"Input depth ({T}) doesn't match model ({self.img_size[0]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCTHW -> BNC
        x = self.norm(x)
        return x