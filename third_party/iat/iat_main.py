"""
Illumination-Adaptive Transformer (IAT) model definition.

Vendored (with small dependency cleanups) from:
  https://github.com/cuiziteng/Illumination-Adaptive-Transformer
  Path: IAT_enhance/model/IAT_main.py

License: Apache-2.0 (see LICENSE in this folder).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import CBlock_ln, SwinTransformerBlock
from .global_net import Global_pred


class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type="ccc"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)
        if type == "ccc":
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == "ttt":
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == "cct":
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        else:
            blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]

        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)
        return mul, add


class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type="ccc"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)
        if type == "ccc":
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == "ttt":
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == "cct":
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        else:
            blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]

        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            try:
                nn.init.trunc_normal_(m.weight, std=0.02)
            except Exception:
                nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add


class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type="lol"):
        super().__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        # Original code used torch.tensordot; matmul is equivalent and more widely supported.
        shape = image.shape
        image = image.view(-1, 3)
        image = image.matmul(ccm.t())
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        if not self.with_global:
            return mul, add, img_high

        gamma, color = self.global_net(img_low)
        b = img_high.shape[0]
        img_high = img_high.permute(0, 2, 3, 1)
        img_high = torch.stack(
            [self.apply_color(img_high[i, :, :, :], color[i, :, :]) ** gamma[i, :] for i in range(b)],
            dim=0,
        )
        img_high = img_high.permute(0, 3, 1, 2)
        return mul, add, img_high

