from __future__ import annotations
from pathlib import Path
from typing import Optional

class ModelManager:
    def __init__(self, models_root: str | Path):
        self.models_root = str(Path(models_root).resolve())

    def exists(self) -> bool:
        return Path(self.models_root).exists()

    def list_top(self) -> list[str]:
        p = Path(self.models_root)
        if not p.exists():
            return []
        return [f.name for f in p.iterdir() if f.is_dir() or f.is_file()]
