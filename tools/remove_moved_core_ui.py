#!/usr/bin/env python3
"""
Remove duplicate / incorrectly-indented CORE UI lines previously moved into setup_gui.

Usage:
  python tools/remove_moved_core_ui.py src/gui/main_window.py

Creates a .bak backup and removes any small blocks that:
 - contain the markers "CORE UI ELEMENTS moved" or "# =============== CORE UI ELEMENTS ============="
 - and the following lines that set up self.time_label / pack / self.pack(...)
"""
from pathlib import Path
import shutil
import sys
import re

TARGET_MARKERS = [
    "CORE UI ELEMENTS moved from class scope",
    "# =============== CORE UI ELEMENTS ============="
]
TIME_LABEL_PAT = re.compile(r"\s*self\.time_label\s*=")
PACK_PAT = re.compile(r"\s*self\.pack\s*\(")

def main(path: Path):
    if not path.exists():
        print("File not found:", path)
        return 1

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"Backup created: {bak}")

    lines = path.read_text(encoding="utf-8").splitlines(True)
    out = []
    i = 0
    removed = 0

    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        # If marker line found, remove marker + following small block that references self.time_label/self.pack
        if any(m in ln for m in TARGET_MARKERS):
            # skip marker line
            i += 1
            # skip small following block lines while they match expected patterns or are blank/comment
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "" or nxt.lstrip().startswith("#"):
                    i += 1
                    continue
                # remove lines that set up time_label or pack
                if TIME_LABEL_PAT.search(nxt) or "time_label.pack" in nxt or PACK_PAT.search(nxt):
                    i += 1
                    continue
                # stop when encountering a normal statement that doesn't look like the stray UI block
                break
            removed += 1
            continue

        # Also remove any stray isolated over-indented time_label blocks that weren't prefixed by marker
        if TIME_LABEL_PAT.search(ln):
            # check a few following lines to decide if this is the stray duplicated block
            lookahead = "".join(lines[i:i+4])
            if ("time_label.pack" in lookahead) or ("self.pack(" in lookahead):
                # skip this small block (time_label line + following up to 3 lines)
                j = i
                while j < len(lines) and j < i + 6:
                    if lines[j].strip() == "" or TIME_LABEL_PAT.search(lines[j]) or "time_label.pack" in lines[j] or PACK_PAT.search(lines[j]) or lines[j].lstrip().startswith("#"):
                        j += 1
                        continue
                    break
                i = j
                removed += 1
                continue

        out.append(ln)
        i += 1

    if removed:
        path.write_text("".join(out), encoding="utf-8")
        print(f"Removed {removed} stray CORE UI block(s) and wrote {path}")
    else:
        print("No stray CORE UI blocks found/removed.")

    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/remove_moved_core_ui.py <target-file>")
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
