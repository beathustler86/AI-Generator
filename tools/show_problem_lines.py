#!/usr/bin/env python3
"""
Print context around a given line number with visible leading whitespace and char codes.
Usage:
  python tools/show_problem_lines.py src/gui/main_window.py 936
"""
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python tools/show_problem_lines.py <file> <lineno> [context=3]")
    sys.exit(2)

path = Path(sys.argv[1])
lineno = int(sys.argv[2])
context = int(sys.argv[3]) if len(sys.argv) > 3 else 3

if not path.exists():
    print("File not found:", path)
    sys.exit(1)

lines = path.read_text(encoding="utf-8").splitlines()
start = max(0, lineno - 1 - context)
end = min(len(lines), lineno + context)

def show_ws(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\t":
            out.append("\\t")
        elif ch == " ":
            out.append("\\u00B7")  # visible middle-dot as escaped codepoint
        elif ord(ch) == 0xA0:
            out.append("\\u00A0")
        else:
            out.append(ch)
    return "".join(out)

print(f"Showing lines {start+1}..{end} from {path}\n")
for i in range(start, end):
    marker = ">>" if (i+1) == lineno else "  "
    raw = lines[i]
    print(f"{marker} {i+1:4d}: {show_ws(raw)}")
    # print first 40 leading char codes for diagnostic
    leading = raw[:min(40, len(raw))]
    codes = " ".join(f"{ord(c):02x}" for c in leading)
    if leading:
        print(f"     leading-chars-hex: {codes}")