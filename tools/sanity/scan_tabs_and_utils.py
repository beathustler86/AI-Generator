import pathlib, re, sys

ROOT = pathlib.Path("src/nodes/cosmos")
TAB_PATTERN = re.compile(r"\t")
DEPRECATED_UTILS_PATTERN = re.compile(r"from\s+\.utils\s+import")
valid_utils_root = pathlib.Path("src/nodes/cosmos/cosmos_tokenizer").resolve()

tab_hits = []
deprecated_hits = []

for py in ROOT.rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="ignore")
    for ln_no, line in enumerate(text.splitlines(), 1):
        if TAB_PATTERN.search(line):
            tab_hits.append(f"{py}:{ln_no}")
        if DEPRECATED_UTILS_PATTERN.search(line):
            if valid_utils_root not in py.resolve().parents and py.resolve() != valid_utils_root:
                deprecated_hits.append(f"{py}:{ln_no}:{line.strip()}")

print(f"Tabs: {len(tab_hits)}")
for h in tab_hits:
    print(h)

print(f"Deprecated utils imports: {len(deprecated_hits)}")
for h in deprecated_hits:
    print(h)

if tab_hits or deprecated_hits:
    sys.exit(1)
print("OK")