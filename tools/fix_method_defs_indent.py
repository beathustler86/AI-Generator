#!/usr/bin/env python3
"""
Ensure all `def` method headers inside class MainWindow have the correct class-level indentation.

Usage:
  python tools/fix_method_defs_indent.py src/gui/main_window.py

Creates a .bak backup before editing. It:
 - locates `class MainWindow`
 - determines the class-level indent from the first non-empty line after the class header
 - for every subsequent line until the next top-level class or EOF, ensures lines that start
   a method (`def `) at class scope are prefixed exactly with the class-level indent.
This fixes mixed/missing indentation that causes "unindent does not match any outer indentation level".
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip("\t "))]

def main(path: Path):
    if not path.exists():
        print("File not found:", path)
        return 1

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"Backup created: {bak}")

    lines = path.read_text(encoding="utf-8").splitlines(True)

    # find class MainWindow
    class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
    if class_idx is None:
        print("class MainWindow not found.")
        return 2

    # determine class body indent from first non-empty line after class header
    class_indent = None
    for i in range(class_idx+1, len(lines)):
        if lines[i].strip() == "":
            continue
        class_indent = leading_ws(lines[i])
        break
    if class_indent is None:
        print("Unable to determine class body indent.")
        return 3

    # scan until next top-level class definition or EOF
    def_region_start = class_idx + 1
    def_region_end = len(lines)
    for i in range(class_idx+1, len(lines)):
        ln = lines[i]
        ws = leading_ws(ln)
        # if we hit another top-level class (no leading ws) then stop
        if ln.lstrip().startswith("class ") and ws == "":
            def_region_end = i
            break

    modified = False
    out = list(lines[:def_region_start])
    i = def_region_start
    while i < def_region_end:
        ln = lines[i]
        stripped = ln.lstrip()
        ws = leading_ws(ln)
        # If this line is a def at class level but has wrong indent, fix it
        if stripped.startswith("def ") or stripped.startswith("@"):
            # For decorator lines (@) we want decorators to have the same indent as the def they decorate.
            # If decorator or def is intended at class level (not deeper), ensure leading equals class_indent.
            # Heuristic: treat a def/decorator as class-level if its current indent length <= len(class_indent) + 1 tab
            # We'll simply make any def/decorator whose ws does NOT start with class_indent use class_indent.
            if not ws == class_indent:
                # preserve trailing newline
                newline = "\n" if ln.endswith("\n") else ""
                content = ln.lstrip()
                out.append(class_indent + content.rstrip("\r\n") + newline)
                modified = True
                i += 1
                continue
        # otherwise append as-is
        out.append(ln)
        i += 1

    # append rest unchanged
    out.extend(lines[def_region_end:])

    if not modified:
        print("No method header indentation fixes required.")
        return 0

    path.write_text("".join(out), encoding="utf-8")
    print("Fixed method header indentation inside MainWindow. Run: python -m py_compile src/gui/main_window.py")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/fix_method_defs_indent.py <target-file>")
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
