#!/usr/bin/env python3
"""
Normalize indentation inside class MainWindow to a consistent tab-based style.

Usage:
  python tools/normalize_class_indentation.py src/gui/main_window.py

Creates a .bak backup before editing. It:
 - finds `class MainWindow(...)`
 - detects the class-body indent and the per-level unit (tab or 4 spaces)
 - normalizes leading whitespace for lines inside the class to use tabs for each indent level
   (keeps class-level indent unchanged).
This reduces "unindent does not match any outer indentation level" caused by mixed tabs/spaces.
"""
from pathlib import Path
import shutil
import sys
import re

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip("\t "))]

def normalize_leading(rest: str, unit_spaces: int) -> str:
    # convert leading spaces groups into tabs, preserve existing tabs
    i = 0
    out = []
    while i < len(rest):
        ch = rest[i]
        if ch == "\t":
            out.append("\t")
            i += 1
            continue
        if ch == " ":
            # count consecutive spaces
            j = i
            while j < len(rest) and rest[j] == " ":
                j += 1
            count = j - i
            tabs = count // unit_spaces
            spaces = count % unit_spaces
            out.append("\t" * tabs)
            out.append(" " * spaces)
            i = j
            continue
        break
    # append the rest (non-leading) unchanged
    out.append(rest[i:])
    return "".join(out)

def main(path: Path):
    if not path.exists():
        print("File not found:", path)
        return 1

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"Backup created: {bak}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(True)

    # find class MainWindow
    class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
    if class_idx is None:
        print("class MainWindow not found.")
        return 2

    # determine class body indent: first non-empty line after class header
    class_indent = None
    for i in range(class_idx+1, len(lines)):
        if lines[i].strip() == "":
            continue
        class_indent = leading_ws(lines[i])
        break
    if class_indent is None:
        print("Unable to determine class body indent.")
        return 3

    # detect per-level unit: find a line inside class with indent deeper than class_indent
    unit_spaces = None
    for i in range(class_idx+1, len(lines)):
        ln = lines[i]
        if ln.strip() == "":
            continue
        ws = leading_ws(ln)
        if ws.startswith(class_indent) and len(ws) > len(class_indent):
            deeper = ws[len(class_indent):]
            # if deeper starts with tab, use 1 tab as unit
            if deeper.startswith("\t"):
                unit_spaces = 1  # special marker for tab unit
                break
            # count spaces
            space_count = len(deeper.replace("\t", ""))
            # if multiple of 4 use 4
            if space_count > 0:
                unit_spaces = 4
                break
    if unit_spaces is None:
        # fallback: assume tabs
        unit_spaces = 1

    # normalize lines inside class until next top-level (class-level) def or EOF
    out_lines = list(lines[:class_idx+1])
    i = class_idx + 1
    while i < len(lines):
        ln = lines[i]
        ws = leading_ws(ln)
        # stop if we hit another top-level class or def at module level
        if ln.lstrip().startswith("class 
