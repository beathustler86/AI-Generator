#!/usr/bin/env python3
"""
Move the small CORE UI ELEMENTS block (class-scope self.* statements)
into the end of the setup_gui method inside MainWindow.

Usage:
    python tools/fix_core_ui_move_to_setup.py src/gui/main_window.py

Creates a .bak backup before editing and prints what it changed.
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
	return s[:len(s) - len(s.lstrip("\t "))]

def main(path: Path):
	if not path.exists():
		print("File not found:", path)
		return 2

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print(f"Backup created: {backup}")

	lines = path.read_text(encoding="utf-8").splitlines(True)

	# find class MainWindow
	class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
	if class_idx is None:
		print("class MainWindow not found.")
		return 3

	# determine class body indent
	class_indent = None
	for i in range(class_idx+1, len(lines)):
		if lines[i].strip() == "":
			continue
		class_indent = leading_ws(lines[i])
		break
	if class_indent is None:
		print("Unable to determine class body indent.")
		return 4

	# find CORE UI marker
	marker = "# =============== CORE UI ELEMENTS ============="
	m_idx = next((i for i,l in enumerate(lines) if l.rstrip("\r\n") == marker), None)
	if m_idx is None:
		print("CORE UI marker not found.")
		return 5

	# Collect following indented lines that look like the stray block (stop at first line that's not indented further)
	block_start = m_idx + 1
	block_end = block_start
	collected = []
	while block_end < len(lines):
		ln = lines[block_end]
		# if line is empty, include it and continue
		if ln.strip() == "":
			collected.append(ln)
			block_end += 1
			continue
		ws = leading_ws(ln)
		# include lines that are indented (ws starts with class_indent + at least one more indent)
		if ws.startswith(class_indent + "\t") or ws.startswith(class_indent + "    "):
			collected.append(ln)
			block_end += 1
			continue
		# otherwise we've reached the end of the stray block
		break

	if not collected:
		print("No indented CORE UI block found after marker.")
		return 6

	# remove the collected block from file
	del lines[block_start:block_end]
	print(f"Removed CORE UI block lines {block_start+1}..{block_end}")

	# find setup_gui inside the class
	setup_idx = None
	for i in range(class_idx+1, len(lines)):
		if lines[i].lstrip().startswith("def setup_gui(") and leading_ws(lines[i]) == class_indent:
			setup_idx = i
			break
	if setup_idx is None:
		print("setup_gui not found at class level.")
		return 7

	# find insertion point: end of setup_gui (next def at class level or end of file)
	insert_at = None
	for i in range(setup_idx+1, len(lines)):
		ws = leading_ws(lines[i])
		if ws == class_indent and lines[i].lstrip().startswith("def "):
			insert_at = i
			break
	if insert_at is None:
		insert_at = len(lines)

	# prepare reindented block for insertion: ensure each non-blank line gets class_indent + one tab
	target_body_indent = class_indent + "\t"
	reindented = []
	reindented.append("\n" + target_body_indent + "# CORE UI ELEMENTS moved from class scope\n")
	for ln in collected:
		if ln.strip() == "":
			reindented.append("\n")
			continue
		# remove leading up to class_indent if present
		if ln.startswith(class_indent):
			content = ln[len(class_indent):].rstrip("\r\n")
		else:
			content = ln.lstrip().rstrip("\r\n")
		reindented.append(target_body_indent + content + "\n")

	# insert reindented block before insert_at
	lines.insert(insert_at, "".join(reindented))
	print(f"Inserted CORE UI block into setup_gui before line {insert_at+1}")

	# write back
	path.write_text("".join(lines), encoding="utf-8")
	print("File updated. Run python -m py_compile src/gui/main_window.py to verify.")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/fix_core_ui_move_to_setup.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))
