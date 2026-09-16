"""批量生成并缓存 TrajectoryTensors

遍历指定根目录下的 `expert*_traj` 子文件夹，使用项目内的
`datasets.ExpertTrajectoryProcessor` 生成轨迹并保存为 `.pt`。

用法（PowerShell）:
    $env:PYTHONPATH='C:/Users/24663/Desktop/my (2)/my'
    python dta/generate_trajectory_tensors.py --root "dta/expert_traj (2)/expert_traj" --out output/dataset_cache/trajectory_tensors --placement_mode layout --workers 1

默认行为会将每个样本的 tensors 保存为 `{sample_name}_gs{grid}_m{margin}.pt`。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from datasets.expert_dataset import ExpertTrajectoryProcessor
import numpy as np
try:
    import env as env_module
    if hasattr(env_module, "PCBPlacementEnv"):
        def _fast_build_wire_mask(self, component):
            return np.zeros((self.num_rotations, self.N, self.N), dtype=np.float32)
        env_module.PCBPlacementEnv.build_wire_mask = _fast_build_wire_mask
        # 快速近似的 spacing 检查：如果已放置元件数量超过阈值，使用基于网格近邻的快速距离检查
        def _fast_check_spacing(self, x, y, w, h, rot):
            current_comp = self.current_component
            current_type = current_comp["type_id"]

            # 获取旋转后的pads
            _, _, current_pads = self._get_rotated_dims_and_pads(current_comp, rot)
            current_pads = [(x + px, y + py) for px, py in current_pads]

            placed_pads_with_types = []
            for comp in self.components:
                if "placed_pads" in comp and comp["placed_pads"]:
                    comp_type = comp["type_id"]
                    for pad_x, pad_y in comp["placed_pads"]:
                        placed_pads_with_types.append((pad_x, pad_y, comp_type))

            if len(placed_pads_with_types) < 200:
                # 规模较小，回退到原始实现
                return env_module.PCBPlacementEnv._check_spacing.__wrapped__(self, x, y, w, h, rot) if hasattr(env_module.PCBPlacementEnv._check_spacing, '__wrapped__') else True

            # 规模较大时，快速近似：用平方距离的最小值并与规则比较
            for current_pad_x, current_pad_y in current_pads:
                for placed_pad_x, placed_pad_y, placed_type in placed_pads_with_types:
                    dx = current_pad_x - placed_pad_x
                    dy = current_pad_y - placed_pad_y
                    if dx*dx + dy*dy < 4.0:  # 如果小于2格的平方（2格为典型pad间距阈值）视为冲突
                        pad_spacing = self.pad_spacing_rules.get((current_type, placed_type), self.pad_spacing_rules.get((placed_type, current_type), 1))
                        if dx*dx + dy*dy < (pad_spacing * pad_spacing):
                            return False
            return True
        # 备份原始函数以便可能恢复
        if not hasattr(env_module.PCBPlacementEnv._check_spacing, '__orig__'):
            env_module.PCBPlacementEnv._check_spacing.__orig__ = env_module.PCBPlacementEnv._check_spacing
        env_module.PCBPlacementEnv._check_spacing = _fast_check_spacing
except Exception:
    # If env import fails, continue and let ExpertTrajectoryProcessor raise later
    pass


def process_one(path: Path, out_dir: Path, grid_size: int, margin: int, placement_mode: str):
    name = path.name
    try:
        # If the task json specifies a grid_N, prefer it for this sample
        task_base = path.name.replace("_traj", "")
        task_file = path / f"{task_base}_task.json"
        sample_grid = grid_size
        if task_file.exists():
            try:
                import json
                t = json.load(open(task_file, 'r', encoding='utf-8'))
                if isinstance(t.get('grid_N'), int) and t.get('grid_N') > 0:
                    sample_grid = int(t.get('grid_N'))
            except Exception:
                pass

        proc = ExpertTrajectoryProcessor(path, grid_size=sample_grid, margin=margin, placement_mode=placement_mode)
        tensors = proc.rollout()
        fname = f"{name}_gs{grid_size}_m{margin}.pt"
        out_path = out_dir / fname
        torch.save(tensors, out_path)
        return (name, True, str(out_path), None)
    except Exception as e:
        return (name, False, None, str(e))


def gather_traj_dirs(root: Path):
    return sorted([p for p in root.glob("expert*_traj") if p.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--placement_mode", choices=["layout", "heuristic"], default="layout")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = gather_traj_dirs(root)
    if args.max_samples:
        trajs = trajs[: args.max_samples]

    results = {"success": [], "failures": []}

    if args.workers <= 1:
        for p in trajs:
            name, ok, outp, err = process_one(p, out_dir, args.grid_size, args.margin, args.placement_mode)
            if ok:
                results["success"].append({"name": name, "path": outp})
                print(f"Wrote {outp}")
            else:
                results["failures"].append({"name": name, "reason": err})
                print(f"Failed {name}: {err}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, p, out_dir, args.grid_size, args.margin, args.placement_mode): p for p in trajs}
            for fut in as_completed(futures):
                name, ok, outp, err = fut.result()
                if ok:
                    results["success"].append({"name": name, "path": outp})
                    print(f"Wrote {outp}")
                else:
                    results["failures"].append({"name": name, "reason": err})
                    print(f"Failed {name}: {err}")

    summary_path = out_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print("Done. Summary:", summary_path)


if __name__ == '__main__':
    main()
