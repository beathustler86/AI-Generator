#!/usr/bin/env python3
"""
Fix indentation of the "CORE UI ELEMENTS moved from class scope" block
inserted into setup_gui that causes "unexpected indent".

Usage:
  python tools/fix_core_ui_reindent.py src/gui/main_window.py

Creates a .bak backup before writing.
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip("\t "))]

def main(path: Path):
    if not path.exists():
        print("File not found:", path)
        return 2

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"Backup created: {bak}")

    lines = path.read_text(encoding="utf-8").splitlines(True)

    # find class MainWindow
    class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
    if class_idx is None:
        print("class MainWindow not found.")
        return 3

    # determine class body indent
    class_indent = None
    for i in range(class_idx+1, len(lines)):
        if lines[i].strip() == "":
            continue
        class_indent = leading_ws(lines[i])
        break
    if class_indent is None:
        print("Unable to determine class body indent.")
        return 4

    # find marker inserted earlier
    marker = "# CORE UI ELEMENTS moved from class scope"
    marker_idx = next((i for i,l in enumerate(lines) if marker in l), None)
    if marker_idx is None:
        print("Marker not found. Nothing to do.")
        return 0

    # collect block lines following marker until next class-level def
    block_start = marker_idx + 1
    block_end = block_start
    while block_end < len(lines):
        ln = lines[block_end]
        ws = leading_ws(ln)
        # stop when we reach a def at class level (same indent as class_indent and starts with def)
        if ws == class_indent and ln.lstrip().startswith("def "):
            break
        block_end += 1

    if block_start >= block_end:
        print("No block found after marker.")
        return 0

    block = lines[block_start:block_end]

    # prepare reindented block: one level deeper than class_indent (match other setup_gui body lines)
    target_indent = class_indent + "\t"
    new_block = []
    new_block.append("\n" + target_indent + "# CORE UI ELEMENTS moved from class scope\n")
    for ln in block:
        if ln.strip() == "":
            new_block.append("\n")
            continue
        # remove any leading whitespace up to class_indent, then prefix target_indent
        if ln.startswith(class_indent):
            content = ln[len(class_indent):].rstrip("\r\n")
        else:
            content = ln.lstrip().rstrip("\r\n")
        # ensure content lines are not over-indented
        # if content already starts with an extra tab, strip one
        if content.startswith("\t"):
            content = content[1:]
        new_block.append(target_indent + content + "\n")

    # replace old block with reindented block
    lines[block_start:block_end] = ["".join(new_block)]

    path.write_text("".join(lines), encoding="utf-8")
    print(f"Reindented CORE UI block lines {block_start+1}..{block_end} -> inserted at same position.")
    print("Run: python -m py_compile src/gui/main_window.py")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/fix_core_ui_reindent.py <target-file>")
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
