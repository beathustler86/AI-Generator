# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import math
import torch
import torch.nn as nn

# Basic sinusoidal timestep embedding
class Timesteps(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) *
                          (torch.arange(half, device=device).float() / max(half-1,1)))
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or hidden_dim
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))

class PatchEmbed(nn.Module):
    """
    Placeholder patch embed for video tensor (B,C,T,H,W) -> (B,T',H',W',D).
    """
    def __init__(self, spatial_patch_size: int, temporal_patch_size: int,
                 in_channels: int, out_channels: int, **_):
        super().__init__()
        self.ps = spatial_patch_size
        self.pt = temporal_patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                              out_channels, bias=False)
    def forward(self, x):
        B,C,T,H,W = x.shape
        assert H % self.ps == 0 and W % self.ps == 0 and T % self.pt == 0
        t_chunks = T // self.pt
        h_chunks = H // self.ps
        w_chunks = W // self.ps
        x = x.view(B, C, t_chunks, self.pt, h_chunks, self.ps, w_chunks, self.ps)
        x = x.permute(0,2,4,6,1,3,5,7).contiguous()
        x = x.view(B, t_chunks, h_chunks, w_chunks, C*self.pt*self.ps*self.ps)
        return self.proj(x)

class GeneralDITTransformerBlock(nn.Module):
    """
    Extremely simplified transformer block placeholder.
    """
    def __init__(self, x_dim: int, **_):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim)
        self.ff = nn.Sequential(
            nn.Linear(x_dim, x_dim*4),
            nn.GELU(),
            nn.Linear(x_dim*4, x_dim)
        )
    def forward(self, x, *args, **kwargs):
        return x + self.ff(self.norm(x))

class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, spatial_patch_size: int,
                 temporal_patch_size: int, out_channels: int, **_):
        super().__init__()
        self.proj = nn.Linear(hidden_size,
                              spatial_patch_size*spatial_patch_size*temporal_patch_size*out_channels)
    def forward(self, x, emb, adaln_lora_B_3D=None):
        return self.proj(x)

__all__ = [
    "Timesteps",
    "TimestepEmbedding",
    "PatchEmbed",
    "GeneralDITTransformerBlock",
    "FinalLayer"
]
