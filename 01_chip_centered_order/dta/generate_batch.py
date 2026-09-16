"""Batch-generate trajectories for a small subset of expert tasks.

This script traverses a directory of expert_*_traj folders, loads the
task JSON for each, uses `pcbagent_repro.PCBEnv` together with the
project `DecisionTransformer` to generate a placement trajectory, and
saves results (or failure reasons) to an output directory.

Run example (PowerShell):
    $env:PYTHONPATH='C:/Users/24663/Desktop/my (2)/my'
    python dta/generate_batch.py --root "dta/expert_traj (2)/expert_traj" --out output/dataset_cache/trajectories --max-samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import List

import numpy as np
import torch


def find_nearest_legal(gx: int, gy: int, rot: int, position_mask: np.ndarray, max_radius: int = 8):
    if position_mask is None:
        return gx, gy, rot
    if position_mask.ndim != 3:
        raise ValueError("position_mask must be 3D (R,N,N)")
    R, N, N2 = position_mask.shape
    gx = int(max(0, min(N - 1, int(round(gx)))))
    gy = int(max(0, min(N2 - 1, int(round(gy)))))
    rot = int(max(0, min(R - 1, int(rot))))
    best = None
    for radius in range(0, max_radius + 1):
        for dx in range(-radius, radius + 1):
            dy = radius - abs(dx)
            for dy_sign in (dy, -dy) if dy != 0 else (dy,):
                x = gx + dx
                y = gy + dy_sign
                if x < 0 or x >= N or y < 0 or y >= N2:
                    continue
                for r in range(R):
                    rot_pref = (r == rot)
                    if position_mask[r, y, x] == 0:
                        rot_diff = 0 if rot_pref else 1
                        score = (rot_diff, abs(dx) + abs(dy_sign))
                        if best is None or score < best[0]:
                            best = (score, x, y, r)
        if best is not None:
            _, bx, by, br = best
            return int(bx), int(by), int(br)
    return gx, gy, rot


def gather_task_dirs(root: str) -> List[str]:
    # find directories containing a *_task.json file
    dirs = []
    for entry in sorted(glob(os.path.join(root, "*"))):
        task_files = glob(os.path.join(entry, "*_task.json"))
        if task_files:
            dirs.append(entry)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    task_dirs = gather_task_dirs(args.root)
    task_dirs = task_dirs[: args.max_samples]

    results = {"success": [], "failures": []}

    # ensure pcbagent_repro path is visible
    pcbdir = os.path.join(os.getcwd(), "pcbagent_repro", "pcbagent_repro")
    if pcbdir not in sys.path:
        sys.path.insert(0, pcbdir)

    try:
        from env import PCBEnv as PCBENv
        from seq import chip_oriented_sequence
        from env import view_mask as pcb_view_mask
        use_pcbenv = True
    except Exception:
        # Fallback: use the repository's PCBPlacementEnv implementation
        try:
            from env import PCBPlacementEnv as PCBENv

            def chip_oriented_sequence(task):
                # Build a simple ordering using PCBPlacementEnv.compute_placement_sequence
                tmp = PCBENv(grid_size=task.get("grid_N", 128), boundary_polygon=task.get("boundary", None),
                              component_list=task.get("components", []), netlist=task.get("nets", []))
                # components in task are dicts with comp_id fields; ensure comp_id present
                comps = task.get("components", [])
                return [c["name"] if "name" in c else f"C{c.get('comp_id', i)}" for i, c in enumerate(comps)]

            def pcb_view_mask(N, placements, comp_index, boundary):
                # Simple view mask based on env.view_mask if placements present
                # placements: dict mapping comp_name->(x,y,rot)
                env_tmp = PCBENv(grid_size=N, boundary_polygon=boundary, component_list=[], netlist=[])
                return env_tmp.view_mask.astype(np.float32)

            use_pcbenv = False
        except Exception as e:
            print("Failed to locate any PCB Env implementation:", e)
            print("Make sure pcbagent_repro exists under project root or repo provides env.PCBPlacementEnv. Aborting.")
            return

    for d in task_dirs:
        name = os.path.basename(d)
        try:
            task_file = glob(os.path.join(d, "*_task.json"))[0]
        except Exception:
            results["failures"].append({"name": name, "reason": "no task file"})
            continue

        try:
            task = json.load(open(task_file, "r", encoding="utf-8"))
        except Exception as e:
            results["failures"].append({"name": name, "reason": f"task load error: {e}"})
            continue

        try:
            env = PCBENv(task)
            env.reset()
            N = env.N
            device = torch.device("cpu")
            # instantiate a model per-task (grid size may vary)
            from DT.models.dt_model import DecisionTransformer

            model = DecisionTransformer(grid_size=N, hidden_size=128)
            model.to(device)
            model.eval()

            seq = chip_oriented_sequence(task)
            traj = []
            for comp_name in seq:
                tokens = env.state_token(comp_name)
                view = pcb_view_mask(env.N, env.placements, env.comp_index, env.boundary).astype(np.float32)
                pos_list = [tokens[r]["position"] for r in sorted(tokens.keys())]
                wire_list = [tokens[r]["wire"] for r in sorted(tokens.keys())]
                pos_stack = np.stack(pos_list, axis=0)
                wire_stack = np.stack(wire_list, axis=0)
                legal_any = (pos_stack == 0).any(axis=0).astype(np.float32)
                wire_avg = wire_stack.mean(axis=0).astype(np.float32)
                denom = float(np.max(np.abs(wire_avg)))
                if denom > 1e-6:
                    wire_avg = wire_avg / denom
                state_np = np.stack([view, legal_any, wire_avg, legal_any], axis=0)
                state_t = torch.from_numpy(state_np).float()

                states_seq = state_t.unsqueeze(0)
                actions_seq = torch.zeros((1, 3), dtype=torch.long)
                returns_seq = torch.zeros((1, 1), dtype=torch.float32)
                timesteps_seq = torch.zeros((1,), dtype=torch.long)

                with torch.no_grad():
                    gx, gy, rot = model.get_action(states_seq, actions_seq, None, returns_seq, timesteps_seq, deterministic=args.deterministic)

                pos_mask = np.stack([tokens[r]["position"] for r in sorted(tokens.keys())], axis=0)
                pgx, pgy, prot = find_nearest_legal(int(gx), int(gy), int(rot), pos_mask, max_radius=8)

                # convert grid -> world coordinates
                xmin, ymin = env.boundary[0]
                xmax, ymax = env.boundary[2]
                W = xmax - xmin
                H = ymax - ymin
                x = xmin + (pgx + 0.5) * (W / env.N)
                y = ymin + (pgy + 0.5) * (H / env.N)

                env.place(comp_name, float(x), float(y), int(prot))
                traj.append({"name": comp_name, "gx": int(pgx), "gy": int(pgy), "rot": int(prot)})

            wl, slw = env.score()
            out = {"name": name, "hpwl": float(wl), "slw": float(slw), "traj": traj}
            out_path = os.path.join(args.out, f"{name}.json")
            json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            results["success"].append({"name": name, "out": out_path})
            print(f"Wrote {out_path} (hpwl={wl:.2f})")

        except Exception as e:
            results["failures"].append({"name": name, "reason": str(e)})

    summary_path = os.path.join(args.out, "batch_summary.json")
    json.dump(results, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Batch run complete. Summary:", summary_path)


if __name__ == "__main__":
    main()
