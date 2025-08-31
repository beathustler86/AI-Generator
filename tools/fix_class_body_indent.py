#!/usr/bin/env python3
"""
Ensure the first non-empty line after class MainWindow has class-body indentation.
If it's an unindented comment, prefix it with one tab so scripts that detect __init__
will work reliably.

Usage:
  python tools/fix_class_body_indent.py src/gui/main_window.py
"""
from pathlib import Path
import shutil
import sys

if len(sys.argv) < 2:
	print("Usage: python tools/fix_class_body_indent.py <target-file>")
	sys.exit(2)

path = Path(sys.argv[1])
if not path.exists():
	print("File not found:", path)
	sys.exit(1)

bak = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, bak)
print("Backup created:", bak)

lines = path.read_text(encoding="utf-8").splitlines(True)
out = list(lines)
changed = False

for i, ln in enumerate(lines):
	if ln.lstrip().startswith("class MainWindow"):
		# find next non-empty line
		j = i + 1
		while j < len(lines) and lines[j].strip() == "":
			j += 1
		if j < len(lines):
			next_ln = lines[j]
			# if next line is an unindented comment, indent it with a single tab
			if next_ln.lstrip().startswith("#") and not next_ln.startswith("\t") and not next_ln.startswith(" "):
				out[j] = "\t" + next_ln
				changed = True
				print(f"Indented comment at line {j+1}")
		break

if changed:
	path.write_text("".join(out), encoding="utf-8")
	print("Patched file to fix class-body indentation.")
else:
	print("No change required (either already indented or class not found).")