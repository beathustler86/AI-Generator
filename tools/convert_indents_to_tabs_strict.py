#!/usr/bin/env python3
"""
Strict conversion: replace any leading mix of spaces/tabs with tabs only.
Creates a .bak backup before overwriting.

Usage:
    python tools/convert_indents_to_tabs_strict.py src/gui/main_window.py
"""
import sys
import shutil
from pathlib import Path

INDENT_SIZE = 4  # from .editorconfig

def leading_to_tabs(ws: str) -> str:
	# compute total width (tabs count as INDENT_SIZE)
	total = 0
	for ch in ws:
		if ch == "\t":
			total += INDENT_SIZE
		elif ch == " ":
			total += 1
		else:
			break
	tabs = total // INDENT_SIZE
	# Strict: discard leftover spaces (force exact tab-based indent)
	return "\t" * tabs

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
		i = 0
		while i < len(ln) and ln[i] in (" ", "\t"):
			i += 1
		leading = ln[:i]
		rest = ln[i:]
		nleading = leading_to_tabs(leading)
		if nleading != leading:
			changed = True
		out_lines.append(nleading + rest)

	if changed:
		path.write_text("".join(out_lines), encoding="utf-8")
		print(f"Converted leading indentation to tabs (INDENT_SIZE={INDENT_SIZE}) for: {path}")
	else:
		print("No changes necessary (file already normalized).")
	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/convert_indents_to_tabs_strict.py <target-file>")
		sys.exit(2)
	target = Path(sys.argv[1])
	sys.exit(convert_file(target))
