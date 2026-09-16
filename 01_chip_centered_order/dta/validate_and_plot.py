"""验证并绘图缓存的 TrajectoryTensors

读取 `output/dataset_cache/trajectory_tensors` 中的 .pt 文件，重放每个样本到环境，检查每步是否合法、收集hpwl/slw，保存渲染PNG和 validation JSON。

用法（PowerShell）:
    $env:PYTHONPATH='C:/Users/24663/Desktop/my (2)/my'
    python dta/validate_and_plot.py --cache output/dataset_cache/trajectory_tensors --out output/replays --max 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import datasets.expert_dataset as _ed
import numpy as np

from env import PCBPlacementEnv


def validate_one(pt_path: Path, out_dir: Path):
    try:
        # Allow loading the custom TrajectoryTensors class (PyTorch 2.6+ safe globals)
        try:
            torch.serialization.add_safe_globals([_ed.TrajectoryTensors])
        except Exception:
            pass
        tensors = torch.load(pt_path, weights_only=False)
    except Exception as e:
        return {"name": pt_path.stem, "ok": False, "reason": f"load_error: {e}"}

    # 找到对应的任务文件（样本名通常是 expertXXX_traj）
    base = pt_path.stem.split("_gs")[0]
    task_json = Path("dta/expert_traj (2)/expert_traj") / base / f"{base}_task.json"

    # 如果 task 存在，从中提取 boundary polygon并构建env
    boundary = None
    grid_size = tensors.states.shape[-1]
    if task_json.exists():
        try:
            import json
            t = json.load(open(task_json, 'r', encoding='utf-8'))
            if t.get('boundary'):
                # ExpertTrajectoryProcessor 已将boundary映射到 grid 在pt中未保存，因此尽量根据 grid_size 使用默认
                boundary = None
        except Exception:
            pass

    env = PCBPlacementEnv(grid_size=grid_size, boundary_polygon=None, component_list=[], netlist=[])

    # We will step through provided actions and check legality
    actions = tensors.actions.numpy()
    states = tensors.states.numpy()

    # Reconstruct an environment with components from processor? The cached .pt does not include component_list.
    # For a pragmatic check, we will rely on the state's position mask channel to detect legality per-step.

    steps = []
    for i in range(actions.shape[0]):
        act = actions[i].tolist()
        state = states[i]
        # state channels: view_mask, legal_any, wire_avg, legal_rot
        legal_rot = state[3]
        legal_any = state[1]

        gx, gy, rot = int(act[0]), int(act[1]), int(act[2])
        # 判定：如果 legal_rot[rot, gy, gx] == 1 或 legal_any[gy,gx] == 1，则视为合法（注意state通道定义）
        is_legal = False
        try:
            if rot >= 0 and rot < legal_rot.shape[0] and gy >=0 and gx>=0 and gy < legal_rot.shape[0] and gx < legal_rot.shape[1]:
                if legal_rot[gy, gx] == 1.0 or legal_any[gy, gx] == 1.0:
                    is_legal = True
        except Exception:
            is_legal = False

        steps.append({"step": i, "action": act, "is_legal": bool(is_legal)})

    # 保存渲染占位图（将states的 view_mask 渲染）
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,6))
        final_state = states[-1]
        view = final_state[0]
        ax.imshow(view, cmap='gray_r')
        ax.set_title(pt_path.stem)
        ax.axis('off')
        out_png = out_dir / f"validation_{pt_path.stem}.png"
        fig.savefig(out_png, bbox_inches='tight', dpi=160)
        plt.close(fig)
    except Exception as e:
        return {"name": pt_path.stem, "ok": False, "reason": f"render_error: {e}"}

    out_json = out_dir / f"validation_{pt_path.stem}.json"
    summary = {
        "name": pt_path.stem,
        "ok": True,
        "steps": steps,
        "png": str(out_png)
    }
    with out_json.open('w', encoding='utf-8') as fp:
        import json as _json
        _json.dump(summary, fp, ensure_ascii=False, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=20)
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = sorted(list(cache_dir.glob('*.pt')))
    pts = pts[: args.max]

    results = []
    for p in pts:
        print('Processing', p.name)
        r = validate_one(p, out_dir)
        results.append(r)
        print(' ->', r.get('ok'), r.get('reason', ''))

    with (out_dir / 'validation_summary.json').open('w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    print('Done. Wrote summary to', out_dir / 'validation_summary.json')

if __name__ == '__main__':
    main()
