#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.region_prior import load_region_targets_for_task
from pcbplace.train import (
    ENV_CUDA_KWARG_DEFAULTS,
    build_placement_env_kwargs,
    validate_placement_env_kwargs,
)



def _add_placement_env_args(ap: argparse.ArgumentParser) -> None:
    """Expose every PlacementEnv CUDA-training knob used by step3/train.

    This keeps the semantic/CUDA diagnostics on the same objective, spacing,
    density, module-region, and edge-clearance configuration as training.
    """
    group = ap.add_argument_group("PlacementEnv diagnostic knobs")
    for key, default in ENV_CUDA_KWARG_DEFAULTS.items():
        group.add_argument(f"--{key}", type=float, default=default)


def _ratio(n: int, d: int) -> str:
    return f"{n}/{d} ({(100.0 * n / max(1, d)):.1f}%)"


def _expert_action_for_ref(env: PlacementEnv, ref: str, comp_json_by_ref: dict):
    """Diagnostic-only replay helper.

    Expert coordinates are read from the source JSON, not from PlacementEnv,
    so runtime task objects remain leakage-safe.
    """
    comp_json = comp_json_by_ref.get(ref) or {}
    expert = comp_json.get("expert") if isinstance(comp_json.get("expert"), dict) else None
    if not expert or expert.get("xy_mm") is None:
        return None
    xmin, ymin, _xmax, _ymax = env.task.bbox_mm
    grid = float(env.task.grid_mm)
    x, y = expert["xy_mm"]
    ix = int((float(x) - xmin) // grid)
    iy = int((float(y) - ymin) // grid)
    rot = int(expert.get("rot", 0))
    if rot in env.rotations:
        ri = env.rotations.index(rot)
    else:
        ri = min(range(len(env.rotations)), key=lambda k: abs(int(env.rotations[k]) - rot))
    w, h = env.grid_shape()
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    return (ri, ix, iy)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check semantic labels are visible to PlacementEnv, region prior, and CUDA env.")
    ap.add_argument("--glob", default="data/fast_train_30_midlarge/*.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--cuda-smoke", action="store_true", help="Run one CUDA mask/objective smoke test when CUDA is available.")
    _add_placement_env_args(ap)
    args = ap.parse_args()

    env_kwargs = build_placement_env_kwargs(vars(args))
    validate_placement_env_kwargs(env_kwargs)
    print(f"[env] diagnostics use PlacementEnv kwargs: {env_kwargs}")

    root = Path(args.root).resolve()
    paths = sorted(glob.glob(str(root / args.glob)))
    if not paths:
        raise SystemExit(f"No JSON files matched: {root / args.glob}")

    total = 0
    env_counts = Counter()
    json_counts = Counter()
    self_anchor = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("components", []):
            total += 1
            if c.get("semantic_class"): json_counts["semantic_class"] += 1
            if c.get("region_type"): json_counts["region_type"] += 1
            if c.get("side_preference"): json_counts["side_preference"] += 1
            if c.get("functional_group"): json_counts["functional_group"] += 1
            if c.get("anchor_ref"): json_counts["anchor_ref"] += 1
            if c.get("subzone"): json_counts["subzone"] += 1
            if c.get("critical_neighbors"): json_counts["critical_neighbors"] += 1
            if c.get("same_side_group"): json_counts["same_side_group"] += 1
            if c.get("boundary_order") is not None: json_counts["boundary_order"] += 1
            if c.get("placement_role"): json_counts["placement_role"] += 1
            if c.get("anchor_ref") and c.get("anchor_ref") == c.get("ref"):
                self_anchor.append((Path(path).name, c.get("ref")))

        task = task_from_json(path)
        env = PlacementEnv(task, **env_kwargs)
        for ref in env.refs:
            fg = env._functional_group.get(ref, "")
            if fg not in ("", "free", None):
                env_counts["functional_group_visible"] += 1
            if isinstance(fg, str) and fg.startswith("module:"):
                env_counts["module_functional_group"] += 1
            if getattr(env, "_placement_roles", {}).get(ref, None) not in ("", None):
                env_counts["placement_role_visible"] += 1
            if env._anchor_refs.get(ref):
                env_counts["anchor_ref"] += 1
            if env._subzones.get(ref) not in ("", "free", None):
                env_counts["subzone_nonfree"] += 1
            if env._critical_neighbors.get(ref):
                env_counts["critical_neighbors"] += 1
            if env._same_side_groups.get(ref):
                env_counts["same_side_group"] += 1
            if env._boundary_orders.get(ref) is not None:
                env_counts["boundary_order"] += 1
            if getattr(env, "_placement_roles", {}).get(ref, "member") != "member":
                env_counts["placement_anchor_role"] += 1

        targets = load_region_targets_for_task(path, 6, 6, 0.85, 0.12, 0.28)
        for ref, tgt in targets.items():
            if tgt.get("anchor_ref"):
                env_counts["region_prior_anchor"] += 1
            if tgt.get("critical_neighbors"):
                env_counts["region_prior_critical"] += 1
            if tgt.get("functional_group", "").startswith("module:"):
                env_counts["region_prior_module_group"] += 1

    print(f"Files: {len(paths)}")
    print(f"Components: {total}")
    for k in [
        "semantic_class", "region_type", "side_preference", "functional_group",
        "anchor_ref", "subzone", "critical_neighbors", "same_side_group",
        "boundary_order", "placement_role",
    ]:
        print(f"JSON {k}: {_ratio(json_counts[k], total)}")

    for k in [
        "functional_group_visible", "placement_role_visible", "module_functional_group",
        "anchor_ref", "subzone_nonfree", "critical_neighbors", "same_side_group",
        "boundary_order", "placement_anchor_role", "region_prior_anchor",
        "region_prior_critical", "region_prior_module_group",
    ]:
        print(f"Runtime {k}: {_ratio(env_counts[k], total)}")

    if self_anchor:
        raise SystemExit(f"Self anchors found, first examples: {self_anchor[:5]}")

    if args.cuda_smoke:
        import torch
        if not torch.cuda.is_available():
            print("CUDA smoke skipped: torch.cuda.is_available() is False")
        else:
            from pcbplace.env_cuda import action_mask_and_bias_cuda, objective_delta_mask_cuda, build_context_tokens_cuda
            device = torch.device("cuda")
            task = task_from_json(paths[0])
            env = PlacementEnv(task, **env_kwargs)
            with open(paths[0], "r", encoding="utf-8") as f:
                smoke_data = json.load(f)
            smoke_comp_json_by_ref = {str(c.get("ref")): c for c in smoke_data.get("components", [])}
            # Replay a few expert-like actions so anchor/critical pair terms have
            # placed context, then run CUDA mask/objective/context.  Expert data
            # stays outside PlacementEnv.
            for _ in range(min(3, len(env.sequence))):
                ref = env.current_ref()
                act = _expert_action_for_ref(env, ref, smoke_comp_json_by_ref)
                if act is None:
                    break
                env.step(act, assume_legal=True, return_observation=False, compute_objective=False)
                if env.done():
                    break
            ref = env.current_ref()
            mask, bias = action_mask_and_bias_cuda(env, ref, device)
            obj = objective_delta_mask_cuda(env, ref, device)
            tok = build_context_tokens_cuda(env, ref, device)
            assert mask.is_cuda and bias.is_cuda and obj["total"].is_cuda and tok.is_cuda
            print(f"CUDA smoke ok: ref={ref}, mask={tuple(mask.shape)}, objective={tuple(obj['total'].shape)}, tokens={tuple(tok.shape)}")

    # Coverage thresholds are based on runtime semantics, not raw JSON fields.
    # PlacementEnv may derive functional_group/placement_role from module and
    # placement context, so CI should fail only when training/inference cannot
    # actually see the labels it will use.
    if env_counts["functional_group_visible"] < total:
        raise SystemExit(
            "Missing runtime functional_group labels: "
            f"{_ratio(env_counts['functional_group_visible'], total)} visible via PlacementEnv"
        )
    if env_counts["placement_role_visible"] < total:
        raise SystemExit(
            "Missing runtime placement_role labels: "
            f"{_ratio(env_counts['placement_role_visible'], total)} visible via PlacementEnv"
        )
    if env_counts["anchor_ref"] == 0 or env_counts["critical_neighbors"] == 0:
        raise SystemExit("Semantic relation labels are not visible to PlacementEnv")
    if env_counts["region_prior_anchor"] == 0:
        raise SystemExit("Semantic relation labels are not visible to region prior")

    print("Semantic usage check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
