from contextlib import contextmanager
from time import perf_counter
from .bus import emit_step, emit_meta
from .config import telemetry_enabled
import uuid

def new_run_id() -> str:
    return uuid.uuid4().hex[:12]

@contextmanager
def step(run_id: str, phase: str, **kv):
    if not telemetry_enabled():
        yield
        return
    s = perf_counter()
    try:
        yield
    finally:
        emit_step(run_id, phase, s, perf_counter(), **kv)

def meta(run_id: str, phase: str, **kv):
    if telemetry_enabled():
        emit_meta(run_id, phase, **kv)