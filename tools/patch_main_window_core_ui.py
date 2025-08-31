#!/usr/bin/env python3
"""
Patch src/gui/main_window.py to neutralize a stray indented "CORE UI ELEMENTS" block
that causes "IndentationError: unindent does not match any outer indentation level".

This script:
 - creates a .bak backup
 - locates the marker line "# =============== CORE UI ELEMENTS ============="
 - comments out following indented lines that reference `self.` (so file will parse)
Usage:
  python tools/patch_main_window_core_ui.py src/gui/main_window.py
"""
import sys
from pathlib import Path
import shutil

def patch(path: Path):
	if not path.exists():
		print("File not found:", path)
		return 2

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print("Backup created:", backup)

	lines = path.read_text(encoding="utf-8").splitlines(True)
	changed = False

	for i, ln in enumerate(lines):
		if ln.strip().startswith("# =============== CORE UI ELEMENTS"):
			# scan following lines and comment any indented lines that start with "self." or "self.pack"
			j = i + 1
			while j < len(lines) and (lines[j].startswith("\t") or lines[j].startswith("    ") or lines[j].strip() == ""):
				trim = lines[j].lstrip()
				if trim.startswith("self.") or trim.startswith("self.pack") or trim.startswith("self.time_label"):
					if not lines[j].lstrip().startswith("#"):
						lines[j] = "# " + lines[j]
						changed = True
				j += 1
			# stop after first found marker handling
			break

	if changed:
		path.write_text("".join(lines), encoding="utf-8")
		print("Patched file: stray CORE UI block commented out.")
	else:
		print("No changes made (either already patched or structure not found).")
	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/patch_main_window_core_ui.py <target-file>")
		sys.exit(2)
	sys.exit(patch(Path(sys.argv[1])))
