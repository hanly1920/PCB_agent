from __future__ import annotations

import argparse
import glob
import inspect
import traceback
from pathlib import Path

import torch

import pcbplace.infer as infer_mod
from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.utils import save_json


def infer_layout_cached(
    task_json_path: str,
    *,
    model,
    meta: dict,
    teacher,
    region_cfg,
    device_t: torch.device,
    fast_step: bool = True,
    postprocess: bool = True,
) -> dict:
    """Fast inference using an already-loaded checkpoint/model.

    This mirrors pcbplace.infer.infer_layout(), but avoids reloading the same
    ckpt for every JSON file.
    """
    required = [
        "_env_kwargs_from_ckpt",
        "_action_features",
        "action_mask_cuda",
        "objective_delta_mask_cuda",
        "build_context_tokens",
        "_policy_logits_and_region_prior_fast",
        "build_teacher_distribution_cuda",
        "_unflatten_action",
        "_apply_action_fast",
        "_layout_objective",
        "_postprocess_layout",
    ]
    missing = [name for name in required if not hasattr(infer_mod, name)]
    if missing:
        raise RuntimeError(
            "当前 pcbplace.infer 缺少加速 infer 需要的函数："
            + ", ".join(missing)
            + "\n请先把之前修改版 pcb_autoplace_fast_infer.zip 覆盖到项目目录。"
        )

    task = task_from_json(task_json_path)
    env_kwargs = infer_mod._env_kwargs_from_ckpt(meta)
    env = PlacementEnv(task, **env_kwargs)

    w, h = env.grid_shape()
    R = len(env.rotations)

    feat = infer_mod._action_features(env)
    feat_t = torch.from_numpy(feat).to(device_t)
    region_idx_cache = {}

    while not env.done():
        ref = env.current_ref()

        mask_flat_t = infer_mod.action_mask_cuda(env, ref, device_t).reshape(-1)
        if not bool((mask_flat_t > 0.5).any().item()):
            env.terminated = True
            break

        obj_maps = infer_mod.objective_delta_mask_cuda(env, ref, device_t)
        objective_delta_t = obj_maps["total"].reshape(-1)

        tokens = infer_mod.build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device_t)

        with torch.inference_mode():
            logits, region_prior_t = infer_mod._policy_logits_and_region_prior_fast(
                model,
                env,
                ref,
                tokens_t,
                feat_t,
                region_cfg,
                device_t,
                region_idx_cache,
            )
            logits = logits.masked_fill(mask_flat_t < 0.5, -1e9)

            if teacher.gate_rollout:
                _q, cand = infer_mod.build_teacher_distribution_cuda(
                    mask_flat_t,
                    objective_delta_t,
                    teacher,
                    device=device_t,
                    region_flat=region_prior_t,
                )
                logits = logits.masked_fill(~cand, -1e9)

            a = int(torch.argmax(logits).item())

        action = infer_mod._unflatten_action(a, w, h, R)
        if fast_step:
            ok, info = infer_mod._apply_action_fast(env, action)
            if not ok or info.get("illegal"):
                break
        else:
            _obs2, _r, _done, info = env.step(action)
            if info.get("illegal"):
                break

    raw_placed = dict(env.placed)
    raw_obj = infer_mod._layout_objective(task, raw_placed, env_kwargs) if raw_placed else 0.0

    if postprocess:
        final_placed = infer_mod._postprocess_layout(task, raw_placed, env_kwargs)
        final_obj = infer_mod._layout_objective(task, final_placed, env_kwargs)
    else:
        final_placed = raw_placed
        final_obj = raw_obj

    return {
        "placed": final_placed,
        "placed_raw": raw_placed,
        "objective": float(final_obj),
        "objective_raw": float(raw_obj),
        "postprocess_applied": bool(postprocess),
        "terminated": bool(env.terminated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_glob", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_fast_step", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--start_index", type=int, default=1, help="1-based index in sorted test_glob")
    parser.add_argument("--end_index", type=int, default=0, help="inclusive 1-based index; 0 means all")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(args.test_glob))
    if args.start_index > 1 or args.end_index > 0:
        lo = max(args.start_index, 1)
        hi = args.end_index if args.end_index > 0 else len(paths)
        paths_to_run = [
            p for i, p in enumerate(paths, start=1)
            if lo <= i <= hi
        ]
    else:
        paths_to_run = paths

    if not paths_to_run:
        raise SystemExit("没有匹配到需要 infer 的 JSON。")

    device_t = infer_mod._resolve_device(args.device)
    print(f"[load ckpt once] {args.ckpt} -> {device_t}")
    model, meta, teacher, region_cfg = infer_mod.load_model(args.ckpt, device=str(device_t))

    completed = 0
    skipped = 0
    failed = []

    total = len(paths)
    run_set = set(paths_to_run)
    for index, path in enumerate(paths, start=1):
        if path not in run_set:
            continue

        src = Path(path)
        out_path = out_dir / f"{src.stem}.infer.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] skip existing: {out_path}")
            skipped += 1
            continue

        try:
            with torch.inference_mode():
                result = infer_layout_cached(
                    path,
                    model=model,
                    meta=meta,
                    teacher=teacher,
                    region_cfg=region_cfg,
                    device_t=device_t,
                    fast_step=not args.no_fast_step,
                    postprocess=not args.no_postprocess,
                )
            save_json(result, out_path)
            print(f"[{index}/{total}] {path} -> {out_path} obj={result['objective']:.3f}")
            completed += 1
        except Exception as error:
            print(f"[{index}/{total}] FAILED: {path}: {error}")
            traceback.print_exc()
            failed.append((path, repr(error)))

    print(f"done completed={completed}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("failed files:")
        for path, error in failed:
            print(f"  {path}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
