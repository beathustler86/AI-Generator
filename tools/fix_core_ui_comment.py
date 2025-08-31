#!/usr/bin/env python3
"""
Comment stray indented `self.*` lines under the
"# =============== CORE UI ELEMENTS =============" marker.

Usage:
  python tools/fix_core_ui_comment.py src/gui/main_window.py
"""
import sys
from pathlib import Path
import shutil

if len(sys.argv) < 2:
	print("Usage: python tools/fix_core_ui_comment.py <target-file>")
	sys.exit(2)

path = Path(sys.argv[1])
if not path.exists():
	print("File not found:", path)
	sys.exit(1)

bak = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, bak)
print("Backup created:", bak)

text = path.read_text(encoding="utf-8")
marker = "# =============== CORE UI ELEMENTS ============="

lines = text.splitlines(True)
changed = False
for i, ln in enumerate(lines):
	if ln.rstrip("\r\n") == marker:
		j = i + 1
		# comment following indented lines that reference self.
		while j < len(lines) and (lines[j].startswith("\t") or lines[j].startswith("    ") or lines[j].strip() == ""):
			trim = lines[j].lstrip()
			if trim.startswith("self.") or trim.startswith("self.pack") or trim.startswith("self.time_label"):
				if not lines[j].lstrip().startswith("#"):
					lines[j] = "# " + lines[j]
					changed = True
			j += 1
		break

if changed:
	path.write_text("".join(lines), encoding="utf-8")
	print("Patched: stray CORE UI lines commented out.")
else:
	print("No changes made (marker not found or already patched).")
