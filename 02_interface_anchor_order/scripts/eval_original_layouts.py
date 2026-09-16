#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate expert/original layouts legality (bounds + overlap).

Fixes vs original:
  - Optional rotation sign multiplier (to support KiCad CW/CCW convention mismatch).
  - Keeps output schema compatible (adds rot_sign_used).

Usage:
  python eval_original_layouts.py --glob "data/json/expert*/expert*_layout.json"
  python eval_original_layouts.py --glob "data/json/expert*/expert*_layout.json" --rot_sign -1
"""

import json
import glob
import math
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

def load_json(p: Path):
    return json.load(open(p, "r", encoding="utf-8"))

def rect_corners(xc, yc, w, h, rot_deg):
    """Return 4 corners of a rotated rectangle (CCW positive) centered at (xc,yc)."""
    th = math.radians(rot_deg % 360.0)
    c = math.cos(th)
    s = math.sin(th)
    dx = w / 2.0
    dy = h / 2.0
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    out = []
    for x0, y0 in corners:
        xr = x0 * c - y0 * s
        yr = x0 * s + y0 * c
        out.append((xc + xr, yc + yr))
    return out

def is_out_of_board(poly, bbox):
    xmin, ymin, xmax, ymax = bbox
    for x, y in poly:
        if x < xmin - 1e-6 or x > xmax + 1e-6 or y < ymin - 1e-6 or y > ymax + 1e-6:
            return True
    return False

def sat_overlap(poly1, poly2, eps=1e-9):
    """Separating axis theorem for convex polygons (here always rectangles)."""
    def axes(poly):
        out = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            ax, ay = -ey, ex
            ln = math.hypot(ax, ay)
            if ln > 0:
                ax /= ln
                ay /= ln
                out.append((ax, ay))
        return out

    def proj(poly, ax, ay):
        mn = 1e30
        mx = -1e30
        for x, y in poly:
            v = x * ax + y * ay
            mn = min(mn, v)
            mx = max(mx, v)
        return mn, mx

    for ax, ay in axes(poly1) + axes(poly2):
        a1, a2 = proj(poly1, ax, ay)
        b1, b2 = proj(poly2, ax, ay)
        # "touching" is NOT treated as overlap
        if a2 <= b1 + eps or b2 <= a1 + eps:
            return False
    return True

def resolve_task_and_placements(path: Path):
    """Support:
      A) *_train.json containing both task + placements
      B) *_layout.json containing placed only (then find sibling *_task.json)
    """
    data = load_json(path)

    # Case A: train json (task + placements)
    if isinstance(data, dict) and "task" in data and "placements" in data:
        task = data["task"]
        placements = {}
        for ref, v in data["placements"].items():
            placements[ref] = (float(v[0]), float(v[1]), float(v[2] if len(v) > 2 else 0))
        return task, placements

    # Case B: layout json (placed only) -> find sibling task
    if isinstance(data, dict) and "placed" in data and path.name.endswith("_layout.json"):
        task_path = path.with_name(path.name.replace("_layout.json", "_task.json"))
        if not task_path.exists():
            raise FileNotFoundError(f"Layout has no sibling task json: {task_path}")
        task = load_json(task_path)
        placements = {}
        for ref, v in data["placed"].items():
            placements[ref] = (float(v[0]), float(v[1]), float(v[2] if len(v) > 2 else 0))
        return task, placements

    raise ValueError(f"Unsupported json format: {path}")

def evaluate_one(path: Path, rot_sign: float = 1.0) -> dict:
    task, placements = resolve_task_and_placements(path)

    bbox = tuple(task["board"]["bbox_mm"])
    comps = task["components"]

    comp_info = {}
    for c in comps:
        ref = c["ref"]
        w, h = c.get("size_mm", [1.0, 1.0])
        typ = c.get("type", "misc")
        comp_info[ref] = {"w": float(w), "h": float(h), "type": str(typ)}

    polys = {}
    out_illegal = []
    out_iface = []
    for ref, (x, y, r) in placements.items():
        if ref not in comp_info:
            continue
        w = comp_info[ref]["w"]
        h = comp_info[ref]["h"]
        typ = comp_info[ref]["type"]
        rr = float(r) * float(rot_sign)
        poly = rect_corners(float(x), float(y), w, h, rr)
        polys[ref] = poly

        if is_out_of_board(poly, bbox):
            if typ == "interface":
                out_iface.append(ref)
            else:
                out_illegal.append(ref)

    # overlaps (pairwise)
    refs = sorted(polys.keys())
    overlaps = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a = refs[i]
            b = refs[j]
            if sat_overlap(polys[a], polys[b]):
                overlaps.append([a, b])

    legal = (len(overlaps) == 0 and len(out_illegal) == 0)

    return {
        "file": str(path),
        "board_bbox_mm": list(bbox),
        "n_components": int(len(comps)),
        "n_placed": int(len(refs)),
        "overlaps": overlaps,
        "out_of_bounds_illegal": sorted(out_illegal),
        "out_of_bounds_interface_ok": sorted(out_iface),
        "legal": bool(legal),
        "rot_sign_used": float(rot_sign),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help=r'例如 "data/json/expert*/expert*_train.json" 或 "data/json/expert*/expert*_layout.json"')
    ap.add_argument("--rot_sign", type=float, default=1.0, help="Rotation multiplier. Use -1 for legacy KiCad CW/CCW mismatch outputs.")
    ap.add_argument("--out_jsonl", default="", help=r'可选：输出汇总 jsonl 文件，例如 data/eval/layout_check.jsonl')
    ap.add_argument("--out_dir", default="", help=r'可选：输出每个板子的单独报告到目录，例如 data/eval/reports')
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    all_rows = []
    for p in paths:
        try:
            row = evaluate_one(p, rot_sign=args.rot_sign)
        except Exception as e:
            row = {"file": str(p), "error": repr(e), "legal": False, "rot_sign_used": float(args.rot_sign)}
        all_rows.append(row)

        if "error" in row:
            print(f"[ERR] {p} -> {row['error']}")
            continue

        tag = "OK" if row["legal"] else "BAD"
        print(f"[{tag}] {p.name}  overlaps={len(row['overlaps'])}  oob_illegal={len(row['out_of_bounds_illegal'])}  oob_iface_ok={len(row['out_of_bounds_interface_ok'])}")

        if row["overlaps"]:
            show = row["overlaps"][:20]
            print("  Overlap pairs (first 20): " + ", ".join([f"{a}-{b}" for a, b in show]))
            if len(row["overlaps"]) > 20:
                print(f"  ... ({len(row['overlaps'])-20} more)")

        if row["out_of_bounds_illegal"]:
            print("  Out-of-bounds (illegal, NOT interface): " + ", ".join(row["out_of_bounds_illegal"]))

        if row["out_of_bounds_interface_ok"]:
            print("  Out-of-bounds (interface, allowed): " + ", ".join(row["out_of_bounds_interface_ok"]))

    # write outputs
    if args.out_jsonl:
        outp = Path(args.out_jsonl)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WROTE] {outp}")

    if args.out_dir:
        outd = Path(args.out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        for row in all_rows:
            if "file" not in row:
                continue
            stem = Path(row["file"]).stem
            outp = outd / f"{stem}.check.json"
            outp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WROTE] per-board reports -> {outd}")

if __name__ == "__main__":
    main()
