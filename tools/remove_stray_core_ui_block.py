#!/usr/bin/env python3
"""
Remove/neutralize a stray indented "CORE UI ELEMENTS" block that sits at class scope
and causes "IndentationError: unindent does not match any outer indentation level".

Usage:
  python tools/remove_stray_core_ui_block.py src/gui/main_window.py

The script:
 - Creates a .bak backup
 - Locates the first occurrence of the marker line:
     # =============== CORE UI ELEMENTS =============
 - If the next few lines are indented and reference `self.*`, the script comments them out
   so they no longer produce an indentation/syntax error.
"""
import sys
from pathlib import Path
import shutil
import re

MARKER = "# =============== CORE UI ELEMENTS ============="

def neutralize(path: Path):
	if not path.exists():
		print("File not found:", path)
		return 2

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print(f"Backup created: {backup}")

	text = path.read_text(encoding="utf-8")
	lines = text.splitlines(True)

	out = []
	i = 0
	changed = False
	while i < len(lines):
		out.append(lines[i])
		# find marker
		if lines[i].rstrip("\r\n") == MARKER:
			# peek ahead a few lines to see if there is an indented block referencing self
			j = i + 1
			block_lines = []
			while j < len(lines) and (lines[j].startswith("\t") or lines[j].startswith("    ") or lines[j].strip() == ""):
				block_lines.append(lines[j])
				# stop when we encounter a blank line followed by a non-indented line (end of stray block)
				j += 1
			# check whether block contains "self." in the first few non-empty lines
			first_non_empty = next((ln for ln in block_lines if ln.strip() != ""), "")
			if "self." in first_non_empty:
				# comment out the indented block
				print(f"Neutralizing stray indented block at lines {i+2}..{j}")
				for k, ln in enumerate(block_lines, start=i+1):
					# comment out if not already commented
					if ln.lstrip().startswith("#"):
						out.append(ln)
					else:
						out.append("# " + ln)
				i = j - 1
				changed = True
		i += 1

	new_text = "".join(out)
	if changed:
		path.write_text(new_text, encoding="utf-8")
		print("Stray block neutralized. File updated.")
	else:
		print("No stray CORE UI block detected/changed.")
	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/remove_stray_core_ui_block.py <target-file>")
		sys.exit(2)
	target = Path(sys.argv[1])
	sys.exit(neutralize(target))
