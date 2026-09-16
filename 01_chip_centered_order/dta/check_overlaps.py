"""Check component bounding-box overlaps for task+layout pairs.

Usage:
  $env:PYTHONPATH='C:/Users/24663/Desktop/my (2)/my'
  python dta/check_overlaps.py --samples expert1000_traj expert1002_traj --root "dta/expert_traj (2)/expert_traj"

Outputs per-sample JSON report to stdout and `output/replays/overlap_<sample>.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def rotated_dims(w: float, h: float, rot: int) -> Tuple[float, float]:
    # rot 0/1/2/3: swap dims on odd rotations
    if rot % 2 == 1:
        return h, w
    return w, h


def bbox_from_pose(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    # assume x,y are top-left
    return (x, y, x + w, y + h)


def intersect_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def check_sample(root: Path, sample: str, out_dir: Path):
    sample_dir = root / sample
    task_file = sample_dir / f"{sample.replace('_traj','')}_task.json"
    layout_file = sample_dir / f"{sample.replace('_traj','')}_layout.json"
    report = {"sample": sample, "status": "ok", "overlaps": []}

    if not task_file.exists() or not layout_file.exists():
        report["status"] = "missing_files"
        return report

    task = load_json(task_file)
    layout = load_json(layout_file)

    comps = task.get('components', [])
    name_to_comp = {c.get('name'): c for c in comps}

    bboxes = []  # list of (name, bbox)
    for name, pose in layout.items():
        comp = name_to_comp.get(name)
        if comp is None:
            continue
        rot = int(round(pose.get('rot', 0) / 90.0)) % 4
        w = float(comp.get('w', 1.0))
        h = float(comp.get('h', 1.0))
        rw, rh = rotated_dims(w, h, rot)
        x = float(pose.get('x', 0.0))
        y = float(pose.get('y', 0.0))
        bbox = bbox_from_pose(x, y, rw, rh)
        bboxes.append((name, bbox))

    # pairwise overlap
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            name_i, bi = bboxes[i]
            name_j, bj = bboxes[j]
            area = intersect_area(bi, bj)
            if area > 0.0:
                report['status'] = 'overlap'
                report['overlaps'].append({
                    'a': name_i, 'b': name_j, 'area': area, 'bbox_a': bi, 'bbox_b': bj
                })

    out_path = out_dir / f"overlap_{sample}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--out', default='output/replays')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    results = []
    for s in args.samples:
        print('Checking', s)
        r = check_sample(root, s, out_dir)
        results.append(r)
        print(' ->', r['status'], 'overlaps=', len(r.get('overlaps', [])))

    summary = out_dir / 'overlap_summary.json'
    with summary.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('Done. Summary at', summary)


if __name__ == '__main__':
    main()
