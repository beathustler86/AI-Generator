#!/usr/bin/env python3
"""
Move stray instance-call lines (e.g. self.setup_gui(), self.build_prompt_ui(), self.build_button_row())
that appear at class scope into the end of MainWindow.__init__.

Usage:
  python tools/move_stray_calls_to_init.py src/gui/main_window.py

Creates a .bak backup before editing.
"""
from pathlib import Path
import shutil
import sys
import re

TARGET_CALLS = ["self.setup_gui()", "self.build_prompt_ui()", "self.build_button_row()"]

def leading_ws(s: str) -> str:
	return s[:len(s) - len(s.lstrip("\t "))]

def find_block_end(lines, start_idx, class_indent):
	# returns index of first line that starts a def at class level (or len(lines))
	for i in range(start_idx, len(lines)):
		ln = lines[i]
		ws = leading_ws(ln)
		if ws == class_indent and ln.lstrip().startswith("def "):
			return i
	return len(lines)

def main(path: Path):
	if not path.exists():
		print("File not found:", path)
		return 2

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print(f"Backup created: {backup}")

	lines = path.read_text(encoding="utf-8").splitlines(True)

	# find class MainWindow line
	class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
	if class_idx is None:
		print("class MainWindow not found.")
		return 3

	# determine class body indent from next non-empty line
	class_indent = None
	for i in range(class_idx+1, len(lines)):
		if lines[i].strip() == "":
			continue
		class_indent = leading_ws(lines[i])
		break
	if class_indent is None:
		print("Unable to determine class body indent.")
		return 4

	# find __init__ def inside class
	init_idx = None
	for i in range(class_idx+1, len(lines)):
		if lines[i].lstrip().startswith("def __init__(") and leading_ws(lines[i]) == class_indent:
			init_idx = i
			break
	if init_idx is None:
		print("__init__ not found at class level.")
		return 5

	# find end of __init__ (index of next def at class level)
	init_end = find_block_end(lines, init_idx+1, class_indent)

	# collect stray lines (occurrences of target calls) outside any def but within class body
	stray_blocks = []
	i = class_idx + 1
	while i < len(lines):
		ln = lines[i]
		ws = leading_ws(ln)
		# if we hit a new def at class level, skip over its body
		if ws == class_indent and ln.lstrip().startswith("def "):
			# find end of this def
			i = find_block_end(lines, i+1, class_indent)
			continue
		# consider only lines that are indented (i.e. part of class body) but not inside a def
		if ws.startswith(class_indent) and any(tc in ln for tc in TARGET_CALLS):
			# capture contiguous block of lines around this (include nearby commented lines above)
			start = i
			# extend backward to include adjacent blank/comment lines above (but not into defs)
			j = i-1
			while j > class_idx and leading_ws(lines[j]).startswith(class_indent) and (lines[j].strip()=="" or lines[j].lstrip().startswith("#")):
				start = j
				j -= 1
			# extend forward to include consecutive lines that are part of the call block
			end = i+1
			while end < len(lines) and any(tc in lines[end] for tc in TARGET_CALLS):
				end += 1
			stray_blocks.append((start, end))
			i = end
			continue
		i += 1

	if not stray_blocks:
		print("No stray target calls found at class scope.")
		return 0

	# remove stray blocks (from bottom to top to preserve indices) and collect their text
	collected = []
	for start, end in reversed(stray_blocks):
		collected.insert(0, "".join(lines[start:end]))
		del lines[start:end]

	# prepare insertion point at end of __init__ (before the next def)
	# recompute init_end in case lines changed earlier indices
	init_end = find_block_end(lines, init_idx+1, class_indent)
	# insert collected blocks (joined) right before init_end
	insert_text = "\n\t\t# Moved from class-scope into __init__\n" + "\n".join(collected) + "\n"

	# determine indentation inside __init__ (one level deeper than class_indent)
	init_body_indent = class_indent + ("\t" if "\t" in class_indent or True else "\t")
	# ensure inserted lines have appropriate indentation: replace leading class_indent with init_body_indent if present
	def reindent_block(block: str) -> str:
		out = []
		for row in block.splitlines(True):
			if row.strip()=="":
				out.append(row)
			else:
				# replace leading whitespace up to class_indent with init_body_indent
				if row.startswith(class_indent):
					out.append(init_body_indent + row[len(class_indent):])
				else:
					out.append(row)
		return "".join(out)

	reindent_text = reindent_block(insert_text)

	lines.insert(init_end, reindent_text)

	path.write_text("".join(lines), encoding="utf-8")
	print(f"Moved {len(collected)} stray block(s) into __init__ and wrote file.")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/move_stray_calls_to_init.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))
