import argparse, json, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    totals = collections.defaultdict(float)
    with open(args.path,"r",encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("type") == "phase":
                totals[(j.get("run_id","-"), j.get("phase"))] += j.get("duration_ms",0)
    # summarize
    phase_aggr = collections.defaultdict(float)
    for (_, phase), v in totals.items():
        phase_aggr[phase]+=v
    for phase, ms in sorted(phase_aggr.items(), key=lambda x:x[1], reverse=True)[:args.top]:
        print(f"{phase:20s} {ms:10.2f} ms")
if __name__ == "__main__":
    main()