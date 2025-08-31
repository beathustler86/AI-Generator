
#!/usr/bin/env python3
"""
Fix indentation for two common mis-indented blocks inside MainWindow:
 - def view_telemetry(self): ensure nested def update_stats and the update_stats() call
   are correctly indented inside the method.
 - def switch_model_threaded(self, selection): ensure nested def task() is indented
   and the threading.Thread(...) call is inside the outer method.

Usage:
  python tools/fix_view_and_switch_indent.py src/gui/main_window.py

Creates a .bak backup before modifying the file and prints what it changed.
"""
from pathlib import Path
import shutil
import sys

def leading_ws(s: str) -> str:
	return s[:len(s) - len(s.lstrip("\t "))]

def rewrite_method_block(lines, start_idx, class_indent):
	# start_idx points at "def ...(" line (class-level). Return (new_lines, end_idx)
	# We will ensure header line keeps class_indent, body lines get class_indent + "\t"
	i = start_idx
	n = len(lines)
	out = []
	# header
	out.append(class_indent + lines[i].lstrip())
	i += 1
	# body until next class-level def or EOF
	while i < n:
		ln = lines[i]
		ws = leading_ws(ln)
		# stop when we hit another def at class level
		if ln.lstrip().startswith("def ") and ws == class_indent:
			break
		# blank lines remain blank (single newline)
		if ln.strip() == "":
			out.append("\n")
			i += 1
			continue
		# compute content (strip any leading whitespace)
		content = ln.lstrip().rstrip("\r\n")
		# indent body one level deeper than class
		out.append(class_indent + "\t" + content + "\n")
		i += 1
	return out, i

def fix_view_telemetry(lines, class_idx, class_indent):
	# find def view_telemetry at class level
	for i in range(class_idx+1, len(lines)):
		if lines[i].lstrip().startswith("def view_telemetry(") and leading_ws(lines[i]) == class_indent:
			start = i
			new_block, end = rewrite_method_block(lines, start, class_indent)
			if new_block:
				# replace
				lines[start:end] = new_block
				return True
	return False

def fix_switch_model_threaded(lines, class_idx, class_indent):
	for i in range(class_idx+1, len(lines)):
		if lines[i].lstrip().startswith("def switch_model_threaded(") and leading_ws(lines[i]) == class_indent:
			# find def line index and rebuild block
			start = i
			new_block, end = rewrite_method_block(lines, start, class_indent)
			if new_block:
				lines[start:end] = new_block
				return True
	return False

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

	changed = False
	if fix_view_telemetry(lines, class_idx, class_indent):
		print("Fixed indentation of view_telemetry method.")
		changed = True
	if fix_switch_model_threaded(lines, class_idx, class_indent):
		print("Fixed indentation of switch_model_threaded method.")
		changed = True

	if changed:
		path.write_text("".join(lines), encoding="utf-8")
		print("File updated. Run: python -m py_compile src/gui/main_window.py")
	else:
		print("No changes made. No target methods found or already correct.")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/fix_view_and_switch_indent.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))