#!/usr/bin/env python3
"""
Fix mixed-tabs/spaces indentation inside class MainWindow.

Usage:
  python tools/fix_indentation_mixed.py src/gui/main_window.py

Creates a .bak backup and normalizes indentation inside the MainWindow class:
- Keeps the class-level indent as-is.
- Converts groups of 4 spaces in deeper indentation to tabs (one tab per 4 spaces).
This is conservative and only touches leading whitespace inside the class body.
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip("\t "))]

def convert_leading_spaces_to_tabs(leading: str, spaces_per_tab: int = 4) -> str:
    # Replace groups of N spaces with tabs, preserve existing tabs and remaining spaces
    # Repeated replacement to collapse longer runs.
    # Work on the string as sequence of chars from left to right.
    out = []
    i = 0
    while i < len(leading):
        ch = leading[i]
        if ch == "\t":
            out.append("\t")
            i += 1
            continue
        if ch == " ":
            # count consecutive spaces
            j = i
            while j < len(leading) and leading[j] == " ":
                j += 1
            count = j - i
            tabs = count // spaces_per_tab
            rem = count % spaces_per_tab
            out.append("\t" * tabs)
            out.append(" " * rem)
            i = j
            continue
        # unexpected char (shouldn't happen), append and continue
        out.append(ch)
        i += 1
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

    # determine class body indent: first non-empty line after class header that appears to be inside class
    class_indent = None
    for i in range(class_idx+1, len(lines)):
        if lines[i].strip() == "":
            continue
        class_indent = leading_ws(lines[i])
        # If the next visible line is a comment but unindented, keep searching
        if class_indent == "":
            continue
        break
    if class_indent is None:
        print("Unable to determine class body indent.")
        return 3

    out = list(lines[:class_idx+1])

    i = class_idx + 1
    while i < len(lines):
        ln = lines[i]
        # if we reach a top-level (module) line (no leading ws) we treat it as end of class
        if ln.strip() != "" and leading_ws(ln) == "" and (ln.lstrip().startswith("class ") or ln.lstrip().startswith("def ")):
            out.extend(lines[i:])
            break

        # Only normalize lines that are inside the class: those whose leading ws starts with class_indent
        ws = leading_ws(ln)
        if ws.startswith(class_indent) or ln.strip() == "":
            # remove class_indent prefix if present, normalize the remainder leading whitespace
            if ln.strip() == "":
                out.append(ln)
                i += 1
                continue
            rest = ln[len(class_indent):] if ln.startswith(class_indent) else ln.lstrip()
            rest_lead = rest[:len(rest) - len(rest.lstrip("\t "))]
            rest_body = rest[len(rest_lead):]
            new_rest_lead = convert_leading_spaces_to_tabs(rest_lead, spaces_per_tab=4)
            new_ln = class_indent + new_rest_lead + rest_body
            out.append(new_ln)
            i += 1
            continue

        # once a line no longer appears to be part of the class body, append remainder and finish
        out.extend(lines[i:])
        break

    path.write_text("".join(out), encoding="utf-8")
    print("Normalized indentation inside MainWindow. Run: python -m py_compile src/gui/main_window.py")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/fix_indentation_mixed.py <target-file>")
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
