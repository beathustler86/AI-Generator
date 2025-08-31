import os

ENV_FLAG = "AI_TOOL_TELEMETRY"
SAMPLE_HZ = 2
LOG_DIR = "logs"
RING_CAP = 512
JSONL_MAX_MB = 50
JSONL_BACKUPS = 5

def telemetry_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").lower() in ("1","true","on","yes")