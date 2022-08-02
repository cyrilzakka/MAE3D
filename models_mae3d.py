# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/MAE
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from einops import rearrange

from timm.models.vision_transformer import Block
from utils.patch_embed import PatchEmbed3D


class MAE3D(nn.Module):
    """  3D Masked Autoencoder with ViT Backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_frames=16, temp_stride=2,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # 3D-MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim, 
                                        num_frames=num_frames, temp_stride=temp_stride)
        self.temp_stride = temp_stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Separable encoder positional embeddings
        self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # 3D-MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Separable encoder positional embeddings
        self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], decoder_embed_dim))
        self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans * self.temp_stride, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed_class, std=.02)
        torch.nn.init.normal_(self.pos_embed_spatial, std=.02)
        torch.nn.init.normal_(self.pos_embed_temporal, std=.02)
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_class, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_spatial, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_temporal, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify3D(self, imgs):
        """
        imgs: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3 *temp_stride)
        """
        x = rearrange(imgs, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=self.temp_stride, p1=self.patch_embed.patch_size[1], p2=self.patch_embed.patch_size[2])
        x = rearrange(x, 'b n p c -> b n (p c)')
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**2 *3 *temp_stride)
        imgs: (N, 3, T, H, W)
        """
        x = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', p1=self.patch_embed.patch_size[1], p2=self.patch_embed.patch_size[2], c=self.in_chans, h=self.patch_embed.grid_size[1], w=self.patch_embed.grid_size[2])
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
    
        # add pos embed w/o cls token
        pos_embed = self.pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                    torch.repeat_interleave(self.pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
        pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        pos_embed = self.decoder_pos_embed_spatial.repeat(1, self.patch_embed.grid_size[0], 1) + \
                    torch.repeat_interleave(self.decoder_pos_embed_temporal, self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2], dim=1)
        pos_embed = torch.cat([self.decoder_pos_embed_class, pos_embed], 1)
        x = x + pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: (N, L, patch_size**2 *3 *temp_stride)
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify3D(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.90):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # (N, L, patch_size**2 *3 *2)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae3d_vit_base_patch16_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae3d_vit_large_patch16_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae3d_vit_huge_patch14_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae3d_vit_base_patch16 = mae3d_vit_base_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
mae3d_vit_large_patch16 = mae3d_vit_large_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
mae3d_vit_huge_patch14 = mae3d_vit_huge_patch14_dec512d4b  # decoder: 512 dim, 4 blocks
