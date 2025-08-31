#!/usr/bin/env python3
"""
Fix MainWindow __init__ to include the refiner check and GUI bootstrap,
and remove the stray class-level refiner/bootstrap block that causes
IndentationError.

Usage:
  python tools/fix_mainwindow_init_refiner.py src/gui/main_window.py

Creates a .bak backup before writing changes.
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
	return s[:len(s) - len(s.lstrip("\t "))]

def find_block(lines, start_idx, class_indent):
	# find end of method/block: next line with same indent and lstrip().startswith("def ") or "class "
	n = len(lines)
	for i in range(start_idx+1, n):
		if leading_ws(lines[i]) == class_indent and (lines[i].lstrip().startswith("def ") or lines[i].lstrip().startswith("class ")):
			return start_idx, i
	return start_idx, n

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

	# find __init__ at class level
	init_idx = next((i for i in range(class_idx+1, len(lines))
					 if lines[i].lstrip().startswith("def __init__(") and leading_ws(lines[i]) == class_indent), None)
	if init_idx is None:
		print("__init__ not found at class level.")
		return 4

	# find end of __init__
	_, init_end = find_block(lines, init_idx, class_indent)

	# Build replacement __init__ body: preserve header, then a standard init body that keeps original first few lines and appends refiner+bootstrap
	# Extract original header and attempt to preserve the first 3 lines inside init (super(), self.root, self.sd35_sessions) if present.
	orig_init_lines = lines[init_idx:init_end]

	# collect minimal existing init inner lines (after def line) up to first blank or 10 lines to preserve any early setup
	# We'll construct a safe, properly-indented init that ends with try/except and GUI bootstrap.
	new_init = []
	# def line (keep as-is but normalized to class_indent)
	new_init.append(class_indent + orig_init_lines[0].lstrip())

	# default inner preserved content: try to find assignments to self.root and self.sd35_sessions in original block
	preserved = []
	for ln in orig_init_lines[1:6]:
		if "self.root" in ln or "self.sd35_sessions" in ln or "super().__init__" in ln:
			preserved.append(ln.lstrip())
	# ensure we at least set self.root and sd35_sessions if not present
	if not any("self.root" in s for s in preserved):
		preserved.append("self.root = root\n")
	if not any("sd35_sessions" in s for s in preserved):
		preserved.append("self.sd35_sessions = sd35_sessions or {}\n")

	# indent preserved lines one level deeper than class
	for p in preserved:
		new_init.append(class_indent + "\t" + p.rstrip("\r\n") + "\n")

	# append the refiner availability check and GUI bootstrap (properly indented)
	refiner_block = [
		"\n",
		class_indent + "\t# === TRY REFINER AVAILABLE ===\n",
		class_indent + "\ttry:\n",
		class_indent + "\t\tfrom src.modules.refiner_module import REFINER_AVAILABLE\n",
		class_indent + "\t\tstatus_text = \"🧠 Refiner: Active\" if REFINER_AVAILABLE else \"🧪 Refiner: Fallback\"\n",
		class_indent + "\t\tstatus_color = \"green\" if REFINER_AVAILABLE else \"orange\"\n",
		class_indent + "\texcept Exception:\n",
		class_indent + "\t\tstatus_text = \"⚠️ Refiner: Error\"\n",
		class_indent + "\t\tstatus_color = \"red\"\n",
		"\n",
		class_indent + "\t# Build GUI\n",
		class_indent + "\tself.setup_gui()\n",
		class_indent + "\t# Build prompt UI after GUI setup\n",
		class_indent + "\tself.build_prompt_ui()\n",
		"\n"
	]
	new_init.extend(refiner_block)

	# Replace init block
	lines[init_idx:init_end] = new_init
	print(f"Rewrote __init__ lines {init_idx+1}..{init_end}")

	# Now remove the stray class-level refiner/bootstrap block later in file if present.
	# Look for marker "# === TRY REFINER AVAILABLE ===" after the class_idx and remove until the following "self.build_prompt_ui()" inclusive.
	start_rm = next((i for i in range(class_idx+1, len(lines)) if lines[i].lstrip().startswith("# === TRY REFINER AVAILABLE ===")), None)
	if start_rm:
		end_rm = None
		for j in range(start_rm, len(lines)):
			if "self.build_prompt_ui()" in lines[j]:
				end_rm = j+1
				break
		if end_rm:
			del lines[start_rm:end_rm]
			print(f"Removed stray refiner/bootstrap block lines {start_rm+1}..{end_rm}")

	# write back
	path.write_text("".join(lines), encoding="utf-8")
	print("File updated. Run: python -m py_compile src/gui/main_window.py")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/fix_mainwindow_init_refiner.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))
