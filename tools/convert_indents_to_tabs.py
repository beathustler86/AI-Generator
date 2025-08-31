#!/usr/bin/env python3
"""
Convert leading indentation in a file to tabs (based on indent_size from .editorconfig).
Creates a .bak backup before overwriting.

Usage:
    python tools/convert_indents_to_tabs.py src/gui/main_window.py
"""
import sys
import shutil
from pathlib import Path

INDENT_SIZE = 4  # matches your .editorconfig

def normalize_leading_indent(ws: str) -> str:
	# Count total indent width in "spaces"
	total = 0
	for ch in ws:
		if ch == "\t":
			total += INDENT_SIZE
		elif ch == " ":
			total += 1
		else:
			break
	# Convert to tabs + leftover spaces (preserve remainder if any)
	tabs = total // INDENT_SIZE
	spaces = total % INDENT_SIZE
	return "\t" * tabs + " " * spaces

def convert_file(path: Path):
	if not path.exists():
		print(f"File not found: {path}")
		return 1

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print(f"Backup created: {backup}")

	lines = path.read_text(encoding="utf-8").splitlines(True)
	changed = False
	out_lines = []
	for ln in lines:
		# separate leading whitespace from rest
		i = 0
		while i < len(ln) and ln[i] in (" ", "\t"):
			i += 1
		leading = ln[:i]
		rest = ln[i:]
		nleading = normalize_leading_indent(leading)
		if nleading != leading:
			changed = True
		out_lines.append(nleading + rest)

	if changed:
		path.write_text("".join(out_lines), encoding="utf-8")
		print(f"Converted indentation to tabs (indent_size={INDENT_SIZE}) for: {path}")
	else:
		print("No changes necessary (file already normalized).")
	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/convert_indents_to_tabs.py <target-file>")
		sys.exit(2)
	target = Path(sys.argv[1])
	sys.exit(convert_file(target))
