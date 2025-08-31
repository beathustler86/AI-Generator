#!/usr/bin/env python3
"""
Remove duplicated 'self.main_frame = tk.Frame(self)' + pack lines that were
accidentally inserted inside setup_gui and cause indentation errors.

Usage:
  python tools/remove_duplicate_main_frame.py src/gui/main_window.py

Creates a .bak backup before editing.
"""
from pathlib import Path
import shutil
import sys

def main(path: Path):
	if not path.exists():
		print("File not found:", path)
		return 1

	bak = path.with_suffix(path.suffix + ".bak")
	shutil.copy2(path, bak)
	print(f"Backup created: {bak}")

	lines = path.read_text(encoding="utf-8").splitlines(True)

	# find class MainWindow
	class_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("class MainWindow")), None)
	if class_idx is None:
		print("class MainWindow not found.")
		return 2

	# find setup_gui def index
	setup_idx = next((i for i in range(class_idx+1, len(lines)) if lines[i].lstrip().startswith("def setup_gui(")), None)
	if setup_idx is None:
		print("setup_gui not found.")
		return 3

	# find end of setup_gui (next def at same class indent or EOF)
	def leading_ws(s: str) -> str:
		return s[:len(s) - len(s.lstrip("\t "))]
	class_indent = leading_ws(lines[class_idx+1]) if class_idx+1 < len(lines) else "\t"
	end_idx = len(lines)
	for i in range(setup_idx+1, len(lines)):
		if leading_ws(lines[i]) == class_indent and lines[i].lstrip().startswith("def "):
			end_idx = i
			break

	# operate only within setup_gui body
	body = lines[setup_idx+1:end_idx]
	out_body = []
	skipped = 0
	i = 0
	while i < len(body):
		ln = body[i]
		# detect the duplicated two-line block (exact or with surrounding whitespace):
		if ln.strip().startswith("self.main_frame = tk.Frame(self)"):
			# check following non-empty/comment line is pack call to remove
			j = i + 1
			# skip possible blank/comment lines between (rare)
			while j < len(body) and body[j].strip() == "":
				j += 1
			if j < len(body) and "self.main_frame.pack" in body[j]:
				# skip i..j
				print(f"Removing duplicated main_frame block at setup_gui lines ~{setup_idx + 1 + i + 1}..{setup_idx + 1 + j + 1}")
				skipped += 1
				i = j + 1
				continue
		# also guard against the variant using self.main_frame = tk.Frame(self) with different whitespace
		if ln.strip().startswith("self.main_frame = tk.Frame(self)") and i+1 < len(body) and "pack(" in body[i+1] and "main_frame" in body[i+1]:
			print(f"Removing duplicated main_frame block at setup_gui lines ~{setup_idx + 1 + i + 1}..{setup_idx + 1 + i + 2}")
			skipped += 1
			i += 2
			continue

		out_body.append(ln)
		i += 1

	if skipped == 0:
		print("No duplicated main_frame block found inside setup_gui.")
		return 0

	# write back
	new_lines = lines[:setup_idx+1] + out_body + lines[end_idx:]
	path.write_text("".join(new_lines), encoding="utf-8")
	print(f"Removed {skipped} duplicated block(s). File updated.")

	return 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python tools/remove_duplicate_main_frame.py <target-file>")
		sys.exit(2)
	sys.exit(main(Path(sys.argv[1])))
