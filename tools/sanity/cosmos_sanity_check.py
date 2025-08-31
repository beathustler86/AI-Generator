import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # adjust as needed
TARGET_DIR = ROOT / "src" / "nodes" / "cosmos"

TAB_PATTERN = re.compile(r"\t")
GUI_PATTERN = re.compile(r"\b(tkinter|ttk|PySimpleGUI)\b")
LEGACY_IMPORT_PATTERN = re.compile(r"\bfrom\s+nodes\.")

bad_tabs = []
gui_leaks = []
legacy_imports = []

for py in TARGET_DIR.rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="ignore")
    for ln_no, line in enumerate(text.splitlines(), 1):
        if TAB_PATTERN.search(line):
            bad_tabs.append(f"{py}:{ln_no}")
        if GUI_PATTERN.search(line):
            gui_leaks.append(f"{py}:{ln_no}:{line.strip()}")
        if LEGACY_IMPORT_PATTERN.search(line):
            legacy_imports.append(f"{py}:{ln_no}:{line.strip()}")

def report(title, items):
    print(f"\n== {title} ({len(items)}) ==")
    for i in items:
        print(i)

report("Mixed Tabs", bad_tabs)
report("Potential GUI Leakage", gui_leaks)
report("Legacy 'nodes.' Imports", legacy_imports)

if bad_tabs or gui_leaks or legacy_imports:
    sys.exit(1)
print("\nCosmos sanity OK")