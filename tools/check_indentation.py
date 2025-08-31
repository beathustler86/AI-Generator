#!/usr/bin/env python3
# Diagnostic: locate lines with leading spaces, mixed tabs+spaces, or non-breaking spaces
from pathlib import Path
import sys

INDENT_SIZE = 4
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/gui/main_window.py")
if not path.exists():
    print("File not found:", path)
    sys.exit(1)

def repr_ws(s: str) -> str:
    # make whitespace visible: \t => \\t, space => ·, NBSP => \\u00A0
    out = []
    for ch in s:
        if ch == "\t":
            out.append("\\t")
        elif ch == " ":
            out.append("·")
        elif ord(ch) == 0xA0:
            out.append("\\u00A0")
        else:
            out.append(ch)
    return "".join(out)

mixed = []
spaces_only = []
nbsp_lines = []
for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    # get leading whitespace
    j = 0
    while j < len(ln) and ln[j] in (" ", "\t", "\u00A0"):
        j += 1
    leading = ln[:j]
    if leading:
        has_tab = "\t" in leading
        has_space = " " in leading
        has_nbsp = "\u00A0" in leading
        if has_nbsp:
            nbsp_lines.append((i, repr_ws(leading), ln[j:]))
        if has_tab and has_space:
            mixed.append((i, repr_ws(leading), ln[j:]))
        elif has_space and not has_tab:
            spaces_only.append((i, repr_ws(leading), ln[j:]))

if not (mixed or spaces_only or nbsp_lines):
    print("No leading-space / mixed / NBSP issues found.")
else:
    if mixed:
        print("Lines with mixed tabs + spaces (leading):")
        for i, r, rest in mixed[:200]:
            print(f"  {i:4d}: {r!s}  -> {rest[:80]!r}")
    if spaces_only:
        print("\nLines with leading spaces (no tabs):")
        for i, r, rest in spaces_only[:200]:
            print(f"  {i:4d}: {r!s}  -> {rest[:80]!r}")
    if nbsp_lines:
        print("\nLines with non-breaking spaces in leading whitespace:")
        for i, r, rest in nbsp_lines[:200]:
            print(f"  {i:4d}: {r!s}  -> {rest[:80]!r}")

# Optional: show a small context around first problematic line
first = (mixed + spaces_only + nbsp_lines)
if first:
    lineno = first[0][0]
    print("\nContext (±3 lines) around first problem line:", lineno)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, lineno-4)
    end = min(len(lines), lineno+3)
    for n in range(start, end):
        prefix = ">>" if (n+1)==lineno else "  "
        print(f"{prefix} {n+1:4d}: {lines[n].rstrip()}")
