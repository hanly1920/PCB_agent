# scripts/scan_expert_snap_radius.py
from __future__ import annotations
import argparse, glob, json, math, os, sys
from collections import defaultdict

import numpy as np

from pcbplace.utils import load_json
from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference in degrees, in [0,180]."""
    d = abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)
    return d


def best_legal_from_mask(mask_ri: np.ndarray, xmin: float, ymin: float, grid: float,
                        x_ex: float, y_ex: float, rot_pen: float = 0.0):
    """Global nearest legal (ix,iy) for one rotation plane mask_ri [X,Y]."""
    idx = np.argwhere(mask_ri > 0.5)  # (N,2) [ix,iy]
    if idx.size == 0:
        return None
    xs = xmin + (idx[:, 0].astype(np.float32) + 0.5) * grid
    ys = ymin + (idx[:, 1].astype(np.float32) + 0.5) * grid
    dx = xs - float(x_ex)
    dy = ys - float(y_ex)
    dist2 = dx * dx + dy * dy
    score = dist2 + (grid * grid) * 0.25 * float(rot_pen)
    k = int(np.argmin(score))
    return float(score[k]), (int(idx[k, 0]), int(idx[k, 1]))


def min_required_radius(mask: np.ndarray, ix0: int, iy0: int) -> int | None:
    """
    mask: [R, X, Y]
    Returns minimal Chebyshev radius r = min over legal (max(|dx|,|dy|)).
    None if no legal anywhere (all zeros).
    """
    best = None
    R = mask.shape[0]
    for ri in range(R):
        idx = np.argwhere(mask[ri] > 0.5)
        if idx.size == 0:
            continue
        dx = np.abs(idx[:, 0] - ix0)
        dy = np.abs(idx[:, 1] - iy0)
        r = int(np.min(np.maximum(dx, dy)))
        if best is None or r < best:
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", type=str, default="data/seq/expert*.json")
    ap.add_argument("--max_files", type=int, default=10_000)
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--write_json", type=str, default="")  # optional output path
    ap.add_argument("--simulate_radius", type=int, default=-1,
                    help="Optional: simulate step3 local-only selection with this radius (no global fallback).")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.train_glob))
    paths = paths[: args.max_files]
    if not paths:
        print(f"[ERR] No files matched: {args.train_glob}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    per_file = defaultdict(lambda: {
        "steps": 0,
        "max_required_radius_cells": 0,
        "max_best_global_move_mm": 0.0,
        "worst_step": None,
    })

    for p in paths:
        data = load_json(p)
        board = data["board"]
        xmin, ymin, xmax, ymax = board["bbox_mm"]
        grid = float(board.get("grid_mm", 1.0))

        comp = {c["ref"]: c for c in data["components"]}
        task = task_from_json(p)
        env = PlacementEnv(task)

        w_cells, h_cells = env.grid_shape()
        rotations = list(env.rotations)
        R = len(rotations)

        t = 0
        while t < args.max_steps:
            obs = env.observe()
            ref = obs["ref"]
            if ref is None:
                break

            ex = comp[ref].get("expert")
            if ex is None:
                # missing expert placement: stop this file
                break

            x_ex, y_ex = float(ex["xy_mm"][0]), float(ex["xy_mm"][1])
            rot_ex = float(ex.get("rot", 0.0))

            ix0 = int(round((x_ex - xmin) / grid - 0.5))
            iy0 = int(round((y_ex - ymin) / grid - 0.5))
            ix0 = max(0, min(w_cells - 1, ix0))
            iy0 = max(0, min(h_cells - 1, iy0))

            mask = obs["action_mask"]  # [R,X,Y]

            # minimal radius needed (existence)
            req_r = min_required_radius(mask, ix0, iy0)

            # compute global-best (like fallback) for stepping forward
            rot_order = sorted(range(R), key=lambda ri: angle_diff(rot_ex, rotations[ri]))
            best = None
            best_score = 1e30
            for ri in rot_order:
                if float(np.max(mask[ri])) < 0.5:
                    continue
                rot_pen = (angle_diff(rot_ex, rotations[ri]) / 90.0) ** 2
                got = best_legal_from_mask(mask[ri], xmin, ymin, grid, x_ex, y_ex, rot_pen=rot_pen)
                if got is None:
                    continue
                score, (ix, iy) = got
                if score < best_score:
                    best_score = score
                    best = (ri, ix, iy)

            if best is None:
                # truly no legal anywhere (should be rare; indicates env/mask bug or impossible state)
                req_r = None
                row = {
                    "file": p, "t": t, "ref": ref,
                    "required_radius_cells": None,
                    "best_global_move_mm": None,
                    "note": "NO_LEGAL_ANYWHERE"
                }
                all_rows.append(row)
                # can't step forward
                break

            xc = xmin + (best[1] + 0.5) * grid
            yc = ymin + (best[2] + 0.5) * grid
            moved_mm = float(math.hypot(xc - x_ex, yc - y_ex))

            row = {
                "file": p,
                "t": t,
                "ref": ref,
                "ix0": ix0, "iy0": iy0,
                "required_radius_cells": req_r,
                "best_global_action": [int(best[0]), int(best[1]), int(best[2])],
                "best_global_move_mm": moved_mm,
                "grid_mm": grid,
            }
            all_rows.append(row)

            pf = per_file[p]
            pf["steps"] += 1
            if req_r is not None and req_r > pf["max_required_radius_cells"]:
                pf["max_required_radius_cells"] = int(req_r)
                pf["worst_step"] = {"t": t, "ref": ref, "required_radius_cells": int(req_r), "best_global_move_mm": moved_mm}
            if moved_mm > pf["max_best_global_move_mm"]:
                pf["max_best_global_move_mm"] = moved_mm

            # step forward using global-best to keep trajectory as close as possible
            _obs2, _r, _done, info = env.step(best)
            if info.get("illegal"):
                # should not happen; record and stop
                all_rows[-1]["note"] = f"STEP_ILLEGAL:{info}"
                break

            t += 1

    # summarize
    reqs = [r["required_radius_cells"] for r in all_rows if isinstance(r.get("required_radius_cells"), int)]
    moves = [r["best_global_move_mm"] for r in all_rows if isinstance(r.get("best_global_move_mm"), (int, float))]

    def pct(xs, p):
        if not xs:
            return None
        return float(np.percentile(np.array(xs, dtype=np.float64), p))

    print("========== Overall ==========")
    print(f"files: {len(paths)}  rows(steps): {len(all_rows)}")
    if reqs:
        print(f"required_radius_cells: max={max(reqs)}  p95={pct(reqs,95):.1f}  p99={pct(reqs,99):.1f}")
    else:
        print("required_radius_cells: (no valid data)")
    if moves:
        print(f"best_global_move_mm: max={max(moves):.3f}  p95={pct(moves,95):.3f}  p99={pct(moves,99):.3f}")
    else:
        print("best_global_move_mm: (no valid data)")

    # per-file top offenders
    items = []
    for fp, st in per_file.items():
        items.append((st["max_required_radius_cells"], st["max_best_global_move_mm"], fp, st["worst_step"], st["steps"]))
    items.sort(reverse=True)

    print("\n========== Top 20 files by max_required_radius_cells ==========")
    for i, (mxr, mxm, fp, worst, steps) in enumerate(items[:20], 1):
        wtxt = ""
        if worst:
            wtxt = f" worst(t={worst['t']}, ref={worst['ref']}, req_r={worst['required_radius_cells']}, move_mm={worst['best_global_move_mm']:.3f})"
        print(f"{i:02d}. req_r_max={mxr:3d}  move_mm_max={mxm:8.3f}  steps={steps:4d}  file={fp}{wtxt}")

    # optional JSON output
    if args.write_json:
        os.makedirs(os.path.dirname(args.write_json) or ".", exist_ok=True)
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump({
                "train_glob": args.train_glob,
                "overall": {
                    "files": len(paths),
                    "rows": len(all_rows),
                    "required_radius_cells": {
                        "max": int(max(reqs)) if reqs else None,
                        "p95": pct(reqs,95) if reqs else None,
                        "p99": pct(reqs,99) if reqs else None,
                    },
                    "best_global_move_mm": {
                        "max": float(max(moves)) if moves else None,
                        "p95": pct(moves,95) if moves else None,
                        "p99": pct(moves,99) if moves else None,
                    },
                },
                "per_file": {
                    fp: st for fp, st in per_file.items()
                },
                "rows": all_rows,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] wrote report: {args.write_json}")

    # optional: simulate local-only with fixed radius (no fallback)
    if args.simulate_radius is not None and args.simulate_radius >= 0:
        Rfix = int(args.simulate_radius)
        print(f"\n========== Simulate local-only (no fallback), radius={Rfix} ==========")
        failed = 0
        worst_move = 0.0

        for p in paths:
            data = load_json(p)
            board = data["board"]
            xmin, ymin, xmax, ymax = board["bbox_mm"]
            grid = float(board.get("grid_mm", 1.0))
            comp = {c["ref"]: c for c in data["components"]}
            task = task_from_json(p)
            env = PlacementEnv(task)
            w_cells, h_cells = env.grid_shape()
            rotations = list(env.rotations)
            RR = len(rotations)

            ok = True
            t = 0
            while t < args.max_steps:
                obs = env.observe()
                ref = obs["ref"]
                if ref is None:
                    break
                ex = comp[ref].get("expert")
                if ex is None:
                    break

                x_ex, y_ex = float(ex["xy_mm"][0]), float(ex["xy_mm"][1])
                rot_ex = float(ex.get("rot", 0.0))

                ix0 = int(round((x_ex - xmin) / grid - 0.5))
                iy0 = int(round((y_ex - ymin) / grid - 0.5))
                ix0 = max(0, min(w_cells - 1, ix0))
                iy0 = max(0, min(h_cells - 1, iy0))

                mask = obs["action_mask"]
                rot_order = sorted(range(RR), key=lambda ri: angle_diff(rot_ex, rotations[ri]))

                best = None
                best_score = 1e30
                for ri in rot_order:
                    if float(np.max(mask[ri])) < 0.5:
                        continue
                    rot_pen = (angle_diff(rot_ex, rotations[ri]) / 90.0) ** 2

                    # local search up to Rfix (same as step3)
                    for rad in range(0, Rfix + 1):
                        x_lo = max(0, ix0 - rad)
                        x_hi = min(w_cells - 1, ix0 + rad)
                        y_lo = max(0, iy0 - rad)
                        y_hi = min(h_cells - 1, iy0 + rad)
                        for ix in range(x_lo, x_hi + 1):
                            for iy in range(y_lo, y_hi + 1):
                                if mask[ri, ix, iy] < 0.5:
                                    continue
                                xc = xmin + (ix + 0.5) * grid
                                yc = ymin + (iy + 0.5) * grid
                                dist2 = (xc - x_ex) ** 2 + (yc - y_ex) ** 2
                                score = dist2 + (grid * grid) * 0.25 * rot_pen
                                if score < best_score:
                                    best_score = score
                                    best = (ri, ix, iy)

                if best is None:
                    print(f"[FAIL] file={p} t={t} ref={ref} (no legal within radius={Rfix})")
                    ok = False
                    break

                xc = xmin + (best[1] + 0.5) * grid
                yc = ymin + (best[2] + 0.5) * grid
                moved_mm = float(math.hypot(xc - x_ex, yc - y_ex))
                worst_move = max(worst_move, moved_mm)

                _obs2, _r, _done, info = env.step(best)
                if info.get("illegal"):
                    print(f"[FAIL] file={p} t={t} ref={ref} illegal={info}")
                    ok = False
                    break
                t += 1

            if not ok:
                failed += 1

        print(f"simulate_radius={Rfix}: failed_files={failed}/{len(paths)}  worst_move_mm={worst_move:.3f}")


if __name__ == "__main__":
    main()
