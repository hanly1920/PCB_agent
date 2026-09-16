import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save(d, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    task = load(args.task)
    layout = load(args.layout)
    placed = layout.get("placed", {})

    missing = []
    n_ok = 0
    for c in task.get("components", []):
        ref = c.get("ref")
        if ref in placed:
            x, y, rot = placed[ref]
            c["expert"] = {"xy_mm": [float(x), float(y)], "rot": float(rot)}
            n_ok += 1
        else:
            missing.append(ref)

    if args.strict and missing:
        raise SystemExit(f"Missing placement for {len(missing)} comps, e.g. {missing[:10]}")

    task.setdefault("graph", {})
    save(task, args.out)
    print(f"[OK] {args.out} expert={n_ok}/{len(task.get('components', []))} missing={len(missing)}")

if __name__ == "__main__":
    main()
