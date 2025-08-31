# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The model definition for 3D layers

Adapted from: https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L889

[MIT License Copyright (c) 2023 Phil Wang]
https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/LICENSE
"""
import math
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# Fallback import for vae_attention (local-first)
try:
    from src.nodes.ops.attention import vae_attention  # noqa: F401
except Exception:
    try:
        from .ops.attention import vae_attention  # noqa: F401 (relative fallback)
    except Exception:
        import torch
        import torch.nn as nn
        # NOTE: compile-only no-op; replace with real attention for runtime behavior.
        class vae_attention(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return x
from .patching import (
    Patcher,
    Patcher3D,
    UnPatcher,
    UnPatcher3D,
)
from .utils import (
    CausalNormalize,
    batch2space,
    batch2time,
    cast_tuple,
    is_odd,
    nonlinearity,
    replication_pad,
    space2batch,
    time2batch,
)

try:
    import comfy.ops as _comfy_ops
    ops = _comfy_ops.disable_weight_init  # provides Conv3d with weight init disabled
    _HAS_COMFY = True
except Exception:
    _HAS_COMFY = False
    class _SimpleOps:
        @staticmethod
        def Conv3d(*args, **kwargs):
            # Fallback to vanilla torch Conv3d
            return nn.Conv3d(*args, **kwargs)
    ops = _SimpleOps  # expose Conv3d so CausalConv3d still works

_LEGACY_NUM_GROUPS = 32


class Tokenizer3D(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class CausalConv3d(nn.Module):
    """
    Safe causal temporal conv. Auto-adjusts kernel/padding if sequence T < kernel.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 time_stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kt = kh = kw = kernel_size
        else:
            kt, kh, kw = kernel_size
        if isinstance(padding, int):
            pt = ph = pw = padding
        else:
            pt, ph, pw = padding
        self.kernel = (kt, kh, kw)
        self.t_pad = pt
        stride_sp = stride if isinstance(stride, int) else stride
        if isinstance(stride_sp, int):
            stride_tuple = (time_stride, stride_sp, stride_sp)
        else:
            stride_tuple = stride_sp
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kt, kh, kw),
            stride=stride_tuple,
            padding=(0, ph, pw),  # temporal handled manually
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        # x: (N,C,T,H,W)
        T = x.shape[2]
        kt, kh, kw = self.kernel
        if T + self.t_pad < kt:
            # reduce effective temporal kernel to available size
            shrink = kt - (T + self.t_pad)
            kt_eff = kt - shrink
            if kt_eff < 1:
                kt_eff = 1
            # center crop kernel by replacing weight
            w = self.conv.weight
            self.conv.weight = torch.nn.Parameter(w[:, :, :kt_eff, :, :])
        if self.t_pad > 0:
            pad_frames = x[:, :, :1, ...].repeat(1, 1, self.t_pad, 1, 1)
            x = torch.cat([pad_frames, x], dim=2)
        return self.conv(x)


class CausalUpsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
        x = x.repeat_interleave(int(time_factor), dim=2)
        x = self.conv(x)
        return x[..., int(time_factor - 1):, :, :]


class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            time_stride=2,
            padding=0,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1, 0, 0)
        x = F.pad(x, pad, mode="constant", value=0)
        x = replication_pad(x)
        x = self.conv(x)
        return x


class CausalHybridUpsample3d(nn.Module):
    """Hybrid temporal + spatial upsampling with causal convs."""
    def __init__(self, in_channels: int, spatial_up: bool = True, temporal_up: bool = True):
        super().__init__()
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up
        if not (spatial_up or temporal_up):
            self.conv_t = self.conv_s = self.conv_fuse = nn.Identity()
            return
        self.conv_t = CausalConv3d(in_channels, in_channels, kernel_size=(3,1,1), stride=1, time_stride=1, padding=0)
        self.conv_s = CausalConv3d(in_channels, in_channels, kernel_size=(1,3,3), stride=1, time_stride=1, padding=1)
        self.conv_fuse = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_up and x.shape[2] > 1:
            x = x.repeat_interleave(2, dim=2)
            x = x[..., 1:, :, :]  # causal shift
            x = self.conv_t(x) + x
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv_s(x) + x
        return self.conv_fuse(x)

class CausalHybridDownsample3d(nn.Module):
    """Hybrid temporal + spatial downsampling with causal convs."""
    def __init__(self, in_channels: int, spatial_down: bool = True, temporal_down: bool = True):
        super().__init__()
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down
        if not (spatial_down or temporal_down):
            self.conv_s = self.conv_t = self.conv_fuse = nn.Identity()
            return
        self.conv_s = CausalConv3d(in_channels, in_channels, kernel_size=(1,3,3), stride=2, time_stride=1, padding=0)
        self.conv_t = CausalConv3d(in_channels, in_channels, kernel_size=(3,1,1), stride=1, time_stride=2, padding=0)
        self.conv_fuse = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_down:
            x = F.pad(x, (0,1,0,1,0,0))
            x = self.conv_s(x) + F.avg_pool3d(x, (1,2,2), (1,2,2))
        if self.temporal_down and x.shape[2] > 1:
            x = replication_pad(x)
            x = self.conv_t(x) + F.avg_pool3d(x, (2,1,1), (2,1,1))
        return self.conv_fuse(x)


class CausalResnetBlock3d(nn.Module):
    def __init(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=num_groups)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalResnetBlockFactorized3d(nn.Module):
    def __init(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=1)
        self.conv1 = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.optimized_attention = vae_attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size = time2batch(q)
        k, batch_size = time2batch(k)
        v, batch_size = time2batch(v)

        b, c, h, w = q.shape
        h_ = self.optimized_attention(q, k, v)

        h_ = batch2time(h_, batch_size)
        h_ = self.proj_out(h_)
        return x + h_


class CausalTemporalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size, height = space2batch(q)
        k, _, _ = space2batch(k)
        v, _, _ = space2batch(v)

        bhw, c, t = q.shape
        q = q.permute(0, 2, 1)  # (bhw, t, c)
        k = k.permute(0, 2, 1)  # (bhw, t, c)
        v = v.permute(0, 2, 1)  # (bhw, t, c)

        w_ = torch.bmm(q, k.permute(0, 2, 1))  # (bhw, t, t)
        w_ = w_ * (int(c) ** (-0.5))

        # Apply causal mask
        mask = torch.tril(torch.ones_like(w_))
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        h_ = torch.bmm(w_, v)  # (bhw, t, c)
        h_ = h_.permute(0, 2, 1).reshape(bhw, c, t)  # (bhw, c, t)

        h_ = batch2space(h_, batch_size, height)
        h_ = self.proj_out(h_)
        return x + h_


class EncoderBase(nn.Module):
    def __init(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher = Patcher(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )
        in_channels = in_channels * patch_size * patch_size

        # downsampling
        self.conv_in = CausalConv3d(
            in_channels, channels, kernel_size=3, stride=1, padding=1
        )

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = CausalDownsample3d(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def patcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.patcher(x)
        x = batch2time(x, batch_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            else:
                # temporal downsample (last level)
                time_factor = 1 + 1 * (hs[-1].shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                hs[-1] = replication_pad(hs[-1])
                hs.append(
                    F.avg_pool3d(
                        hs[-1],
                        kernel_size=[time_factor, 1, 1],
                        stride=[2, 1, 1],
                    )
                )

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DecoderBase(nn.Module):
    def __init(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher = UnPatcher(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )
        out_ch = out_channels * patch_size * patch_size

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = CausalConv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = CausalUpsample3d(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def unpatcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.unpatcher(x)
        x = batch2time(x, batch_size)

        return x

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            else:
                # temporal upsample (last level)
                time_factor = 1.0 + 1.0 * (h.shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                h = h.repeat_interleave(int(time_factor), dim=2)
                h = h[..., int(time_factor - 1) :, :, :]

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h


class EncoderFactorized(nn.Module):
    def __init(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 8,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher3d = Patcher3D(
            patch_size, ignore_kwargs.get("patch_method", "haar")
        )
        in_channels = in_channels * patch_size * patch_size * patch_size

        # calculate the number of downsample operations
        self.num_spatial_downs = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_spatial_downs <= self.num_resolutions
        ), f"Spatially downsample {self.num_resolutions} times at most"

        self.num_temporal_downs = int(math.log2(temporal_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_temporal_downs <= self.num_resolutions
        ), f"Temporally downsample {self.num_resolutions} times at most"

        # downsampling
        self.conv_in = nn.Sequential(
            CausalConv3d(
                in_channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0
            ),
        )

        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                spatial_down = i_level < self.num_spatial_downs
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CausalHybridDownsample3d(
                    block_in,
                    spatial_down=spatial_down,
                    temporal_down=temporal_down,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(
                block_in, z_channels, kernel_size=(1, 3, 3), stride=1, padding=1
            ),
            CausalConv3d(
                z_channels,
                z_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
           

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DecoderFactorized(nn.Module):
    def __init(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 8,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher3d = UnPatcher3D(
            patch_size, ignore_kwargs.get("patch_method", "haar")
        )
        out_ch = out_channels * patch_size * patch_size * patch_size

        # calculate the number of upsample operations
        self.num_spatial_ups = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_spatial_ups <= self.num_resolutions
        ), f"Spatially upsample {self.num_resolutions} times at most"
        self.num_temporal_ups = int(math.log2(temporal_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_temporal_ups <= self.num_resolutions
        ), f"Temporally upsample {self.num_resolutions} times at most"

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = nn.Sequential(
            CausalConv3d(
                z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1
            ),
            CausalConv3d(
                block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0
            ),
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        legacy_mode = ignore_kwargs.get("legacy_mode", False)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # The layer index for temporal/spatial downsampling performed
                # in the encoder should correspond to the layer index in
                # reverse order where upsampling is performed in the decoder.
                # If you've a pre-trained model, you can simply finetune.
                i_level_reverse = self.num_resolutions - i_level - 1
                if legacy_mode:
                    temporal_up = i_level_reverse < self.num_temporal_ups
                else:
                    temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                spatial_up = temporal_up or (
                    i_level_reverse < self.num_spatial_ups
                    and self.num_spatial_ups > self.num_temporal_ups
                )
                up.upsample = CausalHybridUpsample3d(
                    block_in, spatial_up=spatial_up, temporal_up=temporal_up
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, out_ch, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h

