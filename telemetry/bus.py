import queue
from typing import Any, Dict

event_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()

def emit_step(run_id: str, phase: str, start: float, end: float, **kv):
    event_q.put({
        "type": "phase",
        "run_id": run_id,
        "phase": phase,
        "t_start": start,
        "t_end": end,
        "duration_ms": round((end-start)*1000,3),
        **kv
    })

def emit_meta(run_id: str, phase: str, **kv):
    event_q.put({
        "type": "meta",
        "run_id": run_id,
        "phase": phase,
        **kv
    })