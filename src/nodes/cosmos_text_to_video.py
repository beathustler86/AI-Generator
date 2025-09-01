"""
Delegates to cosmos_backend.generate_frames.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

from .cosmos_backend import generate_frames, backend_status

class CosmosTextToVideo:
    def __init__(self):
        st = backend_status()
        if st["backend"] == "stub":
            raise ImportError("Cosmos backend not available (stub mode).")
    def generate(self, prompt: str, frame_count=8, width=512, height=512):
        frames = generate_frames(prompt, frame_count, width, height, seed=0, fps=8, progress_cb=None)
        # Return first frame ndarray for compatibility
        first = np.array(frames[0])
        return first, {"frames": frame_count, "backend": backend_status()}
