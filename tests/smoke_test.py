from __future__ import annotations
import importlib, inspect, sys

def test_blocks_has_timestepembedding() -> None:
    m = importlib.import_module("src.nodes.cosmos.blocks")
    assert hasattr(m, "TimestepEmbedding"), "TimestepEmbedding missing"
    path = inspect.getsourcefile(m)
    assert path and path.replace("\\\\", "\\").endswith("src\\nodes\\cosmos\\blocks.py")

def _try_layers3d_forward():
    try:
        L = importlib.import_module("src.nodes.cosmos.cosmos_tokenizer.layers3d")
    except Exception as e:
        return "skip", f"import failed: {e}"
    if not hasattr(L, "CausalConv3d"):
        return "skip", "CausalConv3d not present"
    try:
        import torch, torch.nn as nn
    except Exception as e:
        return "skip", f"torch not available: {e}"
    try:
        m = nn.Sequential(L.CausalConv3d(3, 4, 3))
        x = torch.randn(1, 3, 2, 8, 8)
        y = m(x)
        return "ok", y.shape
    except Exception as e:
        return "fail", f"forward failed: {e}"

def test_layers3d_dummy_forward():
    status, info = _try_layers3d_forward()
    if status == "ok":
        assert len(info) == 5
    elif status == "skip":
        import pytest
        pytest.skip(info)
    else:
        raise AssertionError(info)

if __name__ == "__main__":
    fails = 0
    try:
        test_blocks_has_timestepembedding()
        print("blocks: OK")
    except Exception as e:
        print("blocks: FAIL", e); fails += 1
    try:
        test_layers3d_dummy_forward()
        print("layers3d: OK or SKIPPED")
    except Exception as e:
        print("layers3d: FAIL", e); fails += 1
    sys.exit(1 if fails else 0)