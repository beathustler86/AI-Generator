#!/usr/bin/env python3
"""
Fix indentation of the generate_video method inside MainWindow.

Usage:
  python tools/fix_generate_video_indent.py src/gui/main_window.py

Creates a .bak backup and normalizes indentation for the whole generate_video
method so nested callback and calls remain inside the method and indentation
units are consistent with the class body indent.
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

	backup = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, backup)
	print(f"Backup created: {backup}")

	lines = path.read_text(encoding="utf-8").splitlines(True)

	# find class MainWindow
	class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
	if class_idx is None:
		print("class MainWindow not found.")
		return 2

	# determine class body indent
	class_indent = None
	for i in range(class_idx+1, len(lines)):
		if lines[i].strip() == "":
			continue
		class_indent = leading_ws(lines[i])
		break
	if class_indent is None:
		print("Unable to determine class body indent.")
		return 3

	# find generate_video at class level
	gen_idx = next((i for i in range(class_idx+1, len(lines))
					if lines[i].lstrip().startswith("def generate_video(") and leading_ws(lines[i]) == class_indent), None)
	if gen_idx is None:
		print("generate_video method not found at class level.")
		return 4

	# find end of method: next def at class level or end of file
	end_idx = len(lines)
	for i in range(gen_idx+1, len(lines)):
		if leading_ws(lines[i]) == class_indent and lines[i].lstrip().startswith("def "):
			end_idx = i
			break

	method_lines = lines[gen_idx:end_idx]

	# Reindent method: def line stays at class_indent, body lines get class_indent + one tab + preserved extra
	new_method = []
	for j, ln in enumerate(method_lines):
		if j == 0:
			# method header
			new_method.append(class_indent + ln.lstrip())
			continue
		if ln.strip() == "":
			new_method.append("\n")
			continue
		ws = leading_ws(ln)
		# determine extra indent beyond class_indent
		if ws.startswith(class_indent):
			extra = ws[len(class_indent):]
		else:
			extra = ws
		# convert groups of 4 spaces in extra to tabs (conservative)
		extra = extra.replace("    ", "\t")
		new_lead = class_indent + "\t" + extra
		content = ln[len(ws):].rstrip("\r\n")
		new_method.append(new_lead + content + "\n")

	# replace in file
	lines[gen_idx:end_idx] = new_method
	path.write_text("".join(lines), encoding="utf-8")
	print(f"Rewrote generate_video method lines {gen_idx+1}..{end_idx}. Run: python -m py_compile {path}")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/fix_generate_video_indent.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))
