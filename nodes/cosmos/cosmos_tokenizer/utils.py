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
"""Shared utilities for the networks module."""

import math
from typing import Any
import torch
from torch import nn
from einops import rearrange  # needed for reshapes

def optimized_attention(q, k, v, heads, mask=None, **_):
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        attn = attn + mask
    attn = attn.softmax(dim=-1)
    return attn @ v

# Simple RMS / Layer-like normalize for embeddings (kept for backward compatibility)
class Normalize(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        mean = x.mean(dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.weight

class CausalNormalize(nn.Module):
    """
    Applies GroupNorm causally across (C,H,W) per time slice.
    If num_groups=1 => LayerNorm-like over channels per frame.
    """
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups=num_groups,
                                 num_channels=in_channels,
                                 eps=1e-6,
                                 affine=True)

    def forward(self, x):
        # x: (B,C,T,H,W)
        if self.num_groups == 1:
            # reshape to apply per-frame GN (effectively LN across C)
            B,C,T,H,W = x.shape
            y = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
            y = self.norm(y)
            return y.view(B, T, C, H, W).permute(0,2,1,3,4)
        return self.norm(x)

# --------- reshape helpers ----------
def time2batch(x: torch.Tensor):
    b = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), b

def batch2time(x: torch.Tensor, batch_size: int):
    t = x.shape[0] // batch_size
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size, t=t)

def space2batch(x: torch.Tensor):
    b, c, t, h, w = x.shape
    return rearrange(x, "b c t h w -> (b h w) c t"), b, h

def batch2space(x: torch.Tensor, batch_size: int, height: int):
    hw = x.shape[0] // batch_size
    width = hw // height
    return rearrange(x, "(b h w) c t -> b c t h w", b=batch_size, h=height, w=width)

def cast_tuple(t: Any, length: int = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def replication_pad(x):
    return torch.cat([x[:, :, :1, ...], x], dim=2)

def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0

def is_odd(n: int) -> bool:
    return not divisible_by(n, 2)

def nonlinearity(x):
    return torch.nn.functional.silu(x)

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_ste(z: torch.Tensor) -> torch.Tensor:
    zhat = z.round()
    return z + (zhat - z).detach()

def log(t, eps=1e-5):
    return t.clamp(min=eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)
