# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import math
import torch
import torch.nn as nn

# Ensure no GUI code remains here (remove any ttk / tk usages)

class TimestepEmbedding(nn.Module):
    """
    Simple two-layer MLP embedding. Adjust dimensions as needed by model import.
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 1024, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or hidden_dim
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, t):
        x = self.lin1(t)
        x = self.act(x)
        return self.lin2(x)

__all__ = [name for name in globals().keys() if name in ("TimestepEmbedding",)]
