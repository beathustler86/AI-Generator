from pathlib import Path
import sys

def leading_ws(s: str) -> str:
    return s[:len(s) - len(s.lstrip("\t "))]

def show_context(path: Path, target: int, ctx: int = 6):
    if not path.exists():
        print("File not found:", path)
        return 2
    lines = path.read_text(encoding="utf-8").splitlines()
    n = len(lines)
    lo = max(1, target - ctx)
    hi = min(n, target + ctx)
    print(f"Showing lines {lo}..{hi} from {path} (total {n} lines)\n")
    for i in range(lo, hi+1):
        ln = lines[i-1]
        ws = leading_ws(ln)
        mixed = (" " in ws) and ("\t" in ws)
        ws_repr = ws.replace("\t", "\\t").replace(" ", "·") or "<none>"
        marker = ">>" if i == target else "  "
        print(f"{marker} {i:5d}: indent={ws_repr:20s} mixed={'YES' if mixed else 'no '} | {ln.rstrip()}")
    # quick scan for mixed indentation in whole file
    mixed_lines = [i+1 for i,l in enumerate(lines) if (" " in leading_ws(l)) and ("\t" in leading_ws(l))]
    if mixed_lines:
        print("\nDetected mixed indentation (tabs+spaces) on lines:", mixed_lines[:20], ("... (more)" if len(mixed_lines)>20 else ""))
    else:
        print("\nNo mixed indentation detected (tabs vs spaces) in leading whitespace.")

    # naive check for unexpected dedent near target:
    def indent_levels(s):
        ws = leading_ws(s)
        # measure as number of tabs + groups of 4 spaces
        tabs = ws.count("\t")
        spaces = ws.count(" ")
        space_levels = spaces // 4
        return tabs + space_levels

    # compute surrounding indent levels
    surrounding = []
    for i in range(max(0, lo-1), min(n, hi)):
        surrounding.append((i+1, indent_levels(lines[i]), lines[i].lstrip()[:60]))
    print("\nSurrounding indent levels (approx):")
    for ln_no, lvl, preview in surrounding:
        flag = "<- target" if ln_no == target else ""
        print(f"  {ln_no:5d}: level={lvl:2d} {flag} | {preview}")

    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_indentation.py <file> [target-line] [context]")
        return 2
    path = Path(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    ctx = int(sys.argv[3]) if len(sys.argv) >= 4 else 6
    return show_context(path, target, ctx)

if __name__ == "__main__":
    sys.exit(main())
