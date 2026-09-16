from __future__ import annotations
import argparse, glob, math, os, sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from pcbplace.train import train, build_placement_env_kwargs, validate_placement_env_kwargs
from pcbplace.utils import load_json
from pcbplace.dataset import task_from_json, SEQUENCE_POLICIES, normalize_sequence_policy
from pcbplace.env import PlacementEnv
from pcbplace.env_cuda import action_mask_and_bias_cuda


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference in degrees, in [0,180]."""
    d = abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)
    return float(d)


def _best_legal_from_mask(mask_ri: np.ndarray, xmin: float, ymin: float, grid: float,
                          x_ex: float, y_ex: float, rot_pen: float = 0.0) -> tuple[float, tuple[int,int]] | None:
    """
    Global nearest legal (ix,iy) for a single rotation plane mask_ri [X,Y].
    Returns (score, (ix,iy)) or None if no legal.
    """
    idx = np.argwhere(mask_ri > 0.5)  # (N,2) with columns [ix,iy]
    if idx.size == 0:
        return None
    xs = xmin + (idx[:, 0].astype(np.float32) + 0.5) * grid
    ys = ymin + (idx[:, 1].astype(np.float32) + 0.5) * grid
    dx = xs - float(x_ex)
    dy = ys - float(y_ex)
    dist2 = dx * dx + dy * dy
    score = dist2 + (grid * grid) * 0.25 * rot_pen
    k = int(np.argmin(score))
    return float(score[k]), (int(idx[k, 0]), int(idx[k, 1]))


def _cuda_action_mask_numpy(env: PlacementEnv, ref: str, device: torch.device) -> np.ndarray:
    """Compute the legal action mask through the CUDA env path, then copy only
    the compact mask back for nearest-expert snapping/validation."""
    mask_t, _bias_t = action_mask_and_bias_cuda(env, ref, device)
    return mask_t.detach().to("cpu").numpy()


def _expert_actions_from_json(task_path: str, snap_radius: int = 6, global_fallback: bool = True, env_kwargs: dict | None = None, device: torch.device | None = None, sequence_policy: str = 'rebuild') -> list[tuple[int,int,int]]:
    """
    Convert expert continuous placements (xy_mm, rot) into discrete (rix, ix, iy) actions.

    Strategy:
      1) Step env in sequence order.
      2) For each component, try to find the closest legal action near rounded (ix0,iy0)
         within Manhattan radius <= snap_radius.
      3) If not found and global_fallback=True, choose the closest legal action anywhere
         in the action_mask (still closest to expert xy/rot).
    """
    data = load_json(task_path)
    board = data["board"]
    xmin, ymin, xmax, ymax = board["bbox_mm"]
    grid = float(board.get("grid_mm", 1.0))

    comp = {c["ref"]: c for c in data["components"]}

    task = task_from_json(task_path, sequence_policy=sequence_policy, load_expert=False)
    env = PlacementEnv(task, **(env_kwargs or {}))
    # task_from_json() applies the explicit sequence_policy selected by the caller.
    seq_len = len(env.sequence)
    if device is None or device.type != 'cuda':
        raise ValueError('step3 expert snapping is CUDA-only; pass a CUDA torch.device')

    w_cells, h_cells = env.grid_shape()
    rotations = list(env.rotations)
    R = len(rotations)

    actions: list[tuple[int,int,int]] = []

    for t in range(seq_len):
        ref = env.current_ref()
        if ref is None:
            break

        ex = comp[ref].get("expert")
        if ex is None:
            raise ValueError(f"Missing expert placement for {ref} in {task_path}")

        x_ex, y_ex = float(ex["xy_mm"][0]), float(ex["xy_mm"][1])
        rot_ex = float(ex.get("rot", 0.0))

        ix0 = int(round((x_ex - xmin) / grid - 0.5))
        iy0 = int(round((y_ex - ymin) / grid - 0.5))
        ix0 = max(0, min(w_cells - 1, ix0))
        iy0 = max(0, min(h_cells - 1, iy0))

        mask = _cuda_action_mask_numpy(env, ref, device)  # [R, X, Y] from CUDA path

        # rotation candidates sorted by closeness to expert rot
        rot_order = sorted(range(R), key=lambda ri: _angle_diff(rot_ex, rotations[ri]))

        best = None
        best_score = 1e30

        # local search first
        for ri in rot_order:
            if float(np.max(mask[ri])) < 0.5:
                continue

            rot_pen = (_angle_diff(rot_ex, rotations[ri]) / 90.0) ** 2

            for rad in range(0, max(0, int(snap_radius)) + 1):
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
                        dist2 = (xc - x_ex) * (xc - x_ex) + (yc - y_ex) * (yc - y_ex)
                        score = dist2 + (grid * grid) * 0.25 * rot_pen
                        if score < best_score:
                            best_score = score
                            best = (ri, ix, iy)

        # global fallback if local fails
        if best is None and global_fallback:
            for ri in rot_order:
                if float(np.max(mask[ri])) < 0.5:
                    continue
                rot_pen = (_angle_diff(rot_ex, rotations[ri]) / 90.0) ** 2
                got = _best_legal_from_mask(mask[ri], xmin, ymin, grid, x_ex, y_ex, rot_pen=rot_pen)
                if got is None:
                    continue
                score, (ix, iy) = got
                if score < best_score:
                    best_score = score
                    best = (ri, ix, iy)

        if best is None:
            c = comp[ref]
            raise ValueError(
                f"No legal action found after discretization. "
                f"file={task_path} t={t} ref={ref} type={c.get('type')} "
                f"allowed_sides={c.get('allowed_sides')} size_mm={c.get('size_mm')} "
                f"expert_xy={ex.get('xy_mm')} expert_rot={rot_ex} ix0={ix0} iy0={iy0} "
                f"snap_radius={snap_radius} global_fallback={global_fallback}"
            )

        # warn if moved far
        xc = xmin + (best[1] + 0.5) * grid
        yc = ymin + (best[2] + 0.5) * grid
        moved = math.hypot(xc - x_ex, yc - y_ex)
        if moved > max(1.0, float(snap_radius) * grid) * 1.5:
            print(f"[WARN] Expert snapped far: file={task_path} t={t} ref={ref} moved_mm={moved:.2f}", file=sys.stderr)

        _obs2, _r, _done, info = env.step(best, assume_legal=True, return_observation=False, compute_objective=False)
        if info.get("illegal"):
            raise ValueError(
                f"Chosen snapped expert action still failed env step. file={task_path} t={t} ref={ref} action={best} info={info}"
            )

        actions.append(best)

    return actions


def _validate_expert(task_path: str, expert_actions: list[tuple[int,int,int]], env_kwargs: dict | None = None, device: torch.device | None = None, sequence_policy: str = 'rebuild') -> None:
    task = task_from_json(task_path, sequence_policy=sequence_policy, load_expert=False)
    env = PlacementEnv(task, **(env_kwargs or {}))
    if device is None or device.type != 'cuda':
        raise ValueError('step3 expert validation is CUDA-only; pass a CUDA torch.device')
    for t, a in enumerate(expert_actions):
        ref = env.current_ref()
        mask = _cuda_action_mask_numpy(env, ref, device)
        rix, ix, iy = a
        ok = (
            (0 <= rix < mask.shape[0])
            and (0 <= ix < mask.shape[1])
            and (0 <= iy < mask.shape[2])
            and (mask[rix, ix, iy] > 0.5)
        )
        if not ok:
            raise ValueError(f"Expert action illegal. file={task_path} t={t} ref={ref} action={a}")
        _obs2, _r, _done, info = env.step(a, assume_legal=True, return_observation=False, compute_objective=False)
        if info.get("illegal"):
            raise ValueError(f"Expert action failed env step. file={task_path} t={t} ref={ref} action={a} info={info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", required=True, help="Glob of training JSON files with expert placements.")
    ap.add_argument(
        "--sequence_policy", default="rebuild", choices=SEQUENCE_POLICIES,
        help="rebuild ignores graph.sequence and computes the canonical order; stored uses graph.sequence exactly; validate requires graph.sequence to equal the canonical rebuilt order.",
    )
    ap.add_argument("--steps", type=int, default=0, help="Main-phase step cap. <=0 means auto until all boards graduate.")
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--kmax_final", type=int, default=-1, help="<=0 means use Tmax so every board can reach full-board training.")
    ap.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate. Fresh training defaults to 3e-4. During --resume, omit this flag to restore the checkpoint optimizer lr; provide it to override the restored lr.",
    )
    ap.add_argument("--seed", type=int, default=7, help="Random seed passed into train().")
    ap.add_argument("--save_path", default="model.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_tokens", type=int, default=128, help="Maximum context-token length passed into train().")
    ap.add_argument("--model_d_model", type=int, default=256,
                    help="Transformer embedding width. Must be divisible by --model_nhead.")
    ap.add_argument("--model_nhead", type=int, default=8,
                    help="Number of Transformer attention heads.")
    ap.add_argument("--model_num_layers", type=int, default=6,
                    help="Number of Transformer encoder layers.")
    ap.add_argument("--model_dropout", type=float, default=0.1,
                    help="Transformer/head dropout in [0,1).")
    ap.add_argument("--min_spacing_mm", type=float, default=0.2,
                    help="Hard spacing margin passed into PlacementEnv for train/expert-snap/infer checkpoint config.")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Number of active boards to process together in the main training phase. >1 enables batched multi-board GPU forward/mask/objective computation.")
    ap.add_argument("--checkpoint_path", default=None, help="Path to rolling/latest checkpoint for save/resume.")
    ap.add_argument("--checkpoint_every_steps", type=int, default=0, help="Save a checkpoint every N main steps. 0 disables periodic checkpointing.")
    ap.add_argument("--keep_last_checkpoints", type=int, default=0, help="Keep only the most recent N step checkpoints. 0 keeps all step checkpoints.")
    ap.add_argument("--resume", action="store_true", help="Resume from --checkpoint_path.")
    ap.add_argument(
        "--resume_restore_config", dest="resume_restore_config", action="store_true",
        help="Restore saved train/teacher/region/geometry/environment configuration before constructing the resumed run (default).",
    )
    ap.add_argument(
        "--no_resume_restore_config", dest="resume_restore_config", action="store_false",
        help="Use the current CLI configuration instead of checkpoint configuration during resume.",
    )
    ap.set_defaults(resume_restore_config=True)

    ap.add_argument("--visdom", dest="visdom_enabled", action="store_true",
                    help="Enable optional Visdom scalar visualization during training.")
    ap.add_argument("--no_visdom", dest="visdom_enabled", action="store_false")
    ap.set_defaults(visdom_enabled=False)
    ap.add_argument("--visdom_server", default="http://localhost")
    ap.add_argument("--visdom_port", type=int, default=8097)
    ap.add_argument("--visdom_env", default="pcb_autoplace")
    ap.add_argument("--visdom_prefix", default="train")
    ap.add_argument("--visdom_log_interval", type=int, default=200)

    ap.add_argument("--expert_snap_radius", type=int, default=6, help="Local search radius (in grid cells) to snap expert to a legal action.")
    ap.add_argument("--expert_snap_global_fallback", action="store_true", help="If local search fails, pick nearest legal anywhere in mask.")
    ap.add_argument("--no_expert_snap_global_fallback", dest="expert_snap_global_fallback", action="store_false")
    ap.set_defaults(expert_snap_global_fallback=True)

    ap.add_argument("--teacher_tau", type=float, default=0.5)
    ap.add_argument("--teacher_lambda_region_prior", type=float, default=0.15,
                    help="Weight of the model-predicted region heatmap log-prob inside teacher energy and rollout gating.")
    ap.add_argument("--teacher_lambda_prior_region_heatmap", type=float, default=0.10,
                    help="Weight of the external leakage-safe prior_region_heatmap inside teacher/action prior.")
    ap.add_argument("--teacher_lambda_conn_prior", type=float, default=0.15,
                    help="Action prior weight for connected-centroid proximity inside teacher/inference gating.")
    ap.add_argument("--teacher_lambda_anchor_prior", type=float, default=0.20,
                    help="Action prior weight for placed-anchor proximity.")
    ap.add_argument("--teacher_lambda_module_prior", type=float, default=0.15,
                    help="Action prior weight for module-center/bbox region hint.")
    ap.add_argument("--teacher_lambda_spacing_prior", type=float, default=0.08,
                    help="Action prior weight for local density/soft-spacing penalty.")
    ap.add_argument("--teacher_lambda_edge_prior", type=float, default=0.05,
                    help="Action prior weight for preferred-edge proximity.")
    ap.add_argument("--teacher_metric_weight", type=float, default=0.15,
                    help="Weight of the expected objective loss E_pi[Delta J].")
    ap.add_argument("--objective_hpwl_weight", type=float, default=1.0,
                    help="Weight for plain HPWL in the unified objective.")
    ap.add_argument("--objective_w_hpwl_weight", type=float, default=0.2)
    ap.add_argument("--objective_nslw_weight", type=float, default=0.05)
    ap.add_argument("--objective_region_weight", type=float, default=0.55)
    ap.add_argument("--objective_module_region_weight", type=float, default=0.35)
    ap.add_argument("--objective_module_floorplan_weight", type=float, default=0.0,
                    help="Weight for module-level floorplan objective: module bbox overlap/channel/compact/prior-region penalty.")
    ap.add_argument("--module_region_bias", type=float, default=0.20)
    ap.add_argument("--module_region_margin_mm", type=float, default=2.0)
    ap.add_argument("--module_floorplan_separation_mm", type=float, default=2.0,
                    help="Preferred channel/gap between unrelated module bboxes; soft only, not a legality constraint.")
    ap.add_argument("--module_floorplan_overlap_scale", type=float, default=1.0)
    ap.add_argument("--module_floorplan_channel_scale", type=float, default=0.65)
    ap.add_argument("--module_floorplan_compact_scale", type=float, default=0.20)
    ap.add_argument("--module_floorplan_region_scale", type=float, default=0.60)
    ap.add_argument("--objective_conn_weight", type=float, default=0.50)
    ap.add_argument("--objective_align_weight", type=float, default=0.28)
    ap.add_argument("--objective_group_weight", type=float, default=0.12)
    ap.add_argument("--objective_anchor_weight", type=float, default=0.18)
    ap.add_argument("--objective_boundary_group_weight", type=float, default=0.22)
    ap.add_argument("--objective_pitch_weight", type=float, default=0.22)
    ap.add_argument("--objective_orientation_weight", type=float, default=0.14)
    ap.add_argument("--objective_edge_clearance_weight", type=float, default=0.40)
    ap.add_argument("--objective_interior_weight", type=float, default=0.30)
    ap.add_argument("--objective_density_weight", type=float, default=0.45)
    ap.add_argument("--objective_soft_spacing_weight", type=float, default=0.32)
    ap.add_argument("--objective_neatness_weight", type=float, default=0.12)
    ap.add_argument("--edge_band_ratio", type=float, default=0.12)
    ap.add_argument("--edge_band_center_ratio", type=float, default=0.55)
    ap.add_argument("--soft_spacing_same_group_extra_mm", type=float, default=0.6)
    ap.add_argument("--soft_spacing_cross_group_extra_mm", type=float, default=1.4)
    ap.add_argument("--soft_spacing_large_extra_mm", type=float, default=0.7)
    ap.add_argument("--same_group_density_scale", type=float, default=0.40)
    ap.add_argument("--critical_neighbor_density_scale", type=float, default=0.25)
    ap.add_argument("--anchor_group_density_scale", type=float, default=0.50)
    ap.add_argument("--large_pair_density_scale", type=float, default=1.20)

    ap.add_argument("--region_prior", dest="region_prior_enabled", action="store_true",
                    help="Enable learned heatmap region prior and semantic auxiliary heads.")
    ap.add_argument("--no_region_prior", dest="region_prior_enabled", action="store_false")
    ap.set_defaults(region_prior_enabled=True)
    ap.add_argument("--region_grid_x", type=int, default=6, help="Region heatmap grid width used by prior generation and the region-heatmap head.")
    ap.add_argument("--region_grid_y", type=int, default=6, help="Region heatmap grid height used by prior generation and the region-heatmap head.")
    ap.add_argument("--region_heatmap_sigma_cells", type=float, default=0.85,
                    help="Gaussian smoothing sigma in region-grid cells for expert/prior heatmap targets.")
    ap.add_argument("--legacy_region_zone_edge_ratio", type=float, default=0.12,
                    help="Legacy semantic fallback edge-band ratio; not used by the heatmap head.")
    ap.add_argument("--legacy_region_zone_core_ratio", type=float, default=0.28,
                    help="Legacy semantic fallback core ratio; not used by the heatmap head.")
    ap.add_argument("--region_heatmap_action_prior_weight", type=float, default=0.35,
                    help="Internal weight for predicted heatmap log-probability before teacher_lambda_region_prior.")
    ap.add_argument("--region_aux_heatmap_weight", type=float, default=0.30,
                    help="Auxiliary loss weight for expert_region_heatmap soft CE/KL supervision.")
    ap.add_argument("--region_aux_prior_consistency_weight", type=float, default=0.03,
                    help="Low-weight consistency loss from prior_region_heatmap to the predicted heatmap.")
    ap.add_argument("--region_aux_semantic_weight", type=float, default=0.10,
                    help="Auxiliary loss weight for semantic-class supervision.")
    ap.add_argument("--region_aux_side_weight", type=float, default=0.12,
                    help="Auxiliary loss weight for side-preference classification.")
    ap.add_argument("--region_aux_subzone_weight", type=float, default=0.10,
                    help="Auxiliary loss weight for anchor-subzone classification.")
    ap.add_argument("--region_aux_pairwise_weight", type=float, default=0.10,
                    help="Auxiliary loss weight for pairwise-relation classification.")

    ap.add_argument("--teacher_topk", type=int, default=256)
    ap.add_argument("--teacher_objective_delta_max", type=float, default=None)
    ap.add_argument("--teacher_gate_rollout", dest="teacher_gate_rollout", action="store_true")

    ap.add_argument("--env_alignment_bonus", type=float, default=0.05)
    ap.add_argument("--env_edge_bonus", type=float, default=0.15)
    ap.add_argument("--env_edge_eps_mm", type=float, default=1.5)
    ap.add_argument("--reward_non_interface_edge_penalty", type=float, default=10.0,
                    help="Penalty weight for placing non-interface parts too close to the board edge.")
    ap.add_argument("--reward_non_interface_edge_margin_mm", type=float, default=2.5,
                    help="Desired minimum clearance from board edge for non-interface parts.")
    ap.add_argument("--reward_density_penalty", type=float, default=3.0,
                    help="Penalty weight for local crowding among non-interface parts.")
    ap.add_argument("--reward_density_radius_mm", type=float, default=4.0,
                    help="Extra soft spacing radius beyond part size when computing density penalty.")
    ap.add_argument("--reward_interior_penalty", type=float, default=1.0,
                    help="Penalty weight for placing non-interface parts outside the interior band.")
    ap.add_argument("--reward_interior_margin_ratio", type=float, default=0.18,
                    help="Interior band margin ratio used by the interior regularizer.")
    ap.add_argument("--no_teacher_gate_rollout", dest="teacher_gate_rollout", action="store_false")
    ap.set_defaults(teacher_gate_rollout=True)

    ap.add_argument("--expert_mix_start", type=float, default=1.0)
    ap.add_argument("--expert_mix_end", type=float, default=0.40)
    ap.add_argument("--expert_mix_anneal_steps", type=int, default=80000)

    ap.add_argument("--geometry_xy_weight", type=float, default=0.05)
    ap.add_argument("--geometry_rot_weight", type=float, default=0.02)
    ap.add_argument("--geometry_align_offset_weight", type=float, default=0.03)
    ap.add_argument("--geometry_boundary_axis_weight", type=float, default=0.03)
    ap.add_argument("--geometry_module_bbox_weight", type=float, default=0.0,
                    help="Deprecated no-op. Expert module-bbox distillation was removed from runtime training to avoid label leakage.")

    ap.add_argument("--replay_finetune", dest="replay_finetune", action="store_true")
    ap.add_argument("--no_replay_finetune", dest="replay_finetune", action="store_false")
    ap.set_defaults(replay_finetune=True)
    ap.add_argument("--replay_iters", type=int, default=20)
    ap.add_argument("--replay_rollouts_per_iter", type=int, default=16)
    ap.add_argument("--replay_update_steps", type=int, default=32)
    ap.add_argument("--replay_batch_size", type=int, default=4)
    ap.add_argument("--replay_capacity", type=int, default=2000)
    ap.add_argument("--replay_alpha", type=float, default=0.7)
    ap.add_argument("--replay_temp", type=float, default=1.0)
    ap.add_argument("--replay_action_ce_coef", type=float, default=0.1)
    ap.add_argument(
        "--replay_policy_gradient_coef",
        type=float,
        default=1.0,
        help="REINFORCE loss coefficient for fresh on-policy replay rollouts.",
    )
    ap.add_argument(
        "--replay_entropy_coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient for fresh replay policy-gradient updates.",
    )
    ap.add_argument(
        "--replay_rollout_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used to collect on-policy replay trajectories.",
    )
    ap.add_argument(
        "--replay_baseline_beta",
        type=float,
        default=0.90,
        help="EMA beta for the per-board REINFORCE baseline.",
    )
    ap.add_argument(
        "--replay_pg_policy_mode",
        choices=("shaped", "pure_logits"),
        default="shaped",
        help="REINFORCE log-prob policy: shaped matches rollout/inference scorer; pure_logits uses model logits only for ablation.",
    )
    ap.add_argument(
        "--replay_advantage_clip",
        type=float,
        default=1.0,
        help="Absolute clipping bound for normalized replay advantages; 0 disables.",
    )
    ap.add_argument("--replay_k", type=int, default=None)


    ap.add_argument("--adaptive_k", dest="adaptive_k", action="store_true",
                    help="Adaptive k curriculum: track readiness per board at current k, and only increase k after every unfinished board is ready.")
    ap.add_argument("--no_adaptive_k", dest="adaptive_k", action="store_false")
    ap.set_defaults(adaptive_k=False)
    ap.add_argument("--fixed_k_steps", type=int, default=0,
                    help="When >0 and adaptive_k is enabled, promote k every N global main steps, ignoring per-board target readiness.")
    ap.add_argument(
        "--curriculum_window_mode",
        choices=("suffix", "prefix", "random", "mixed"),
        default="suffix",
        help=(
            "Training window mode. suffix is the old behavior; prefix trains the front-k actions "
            "from an empty board; random trains a random k-step window; mixed uses 50% prefix, "
            "25% suffix, and 25% random windows."
        ),
    )
    ap.add_argument("--adaptive_k_drop_ratio", type=float, default=0.7,
                    help="For k>=2: require EMA loss to drop by this fraction from the peak after switching to current k.")
    ap.add_argument("--adaptive_k_drop_ratio_first", type=float, default=0.4,
                    help="For k==1: require EMA loss to drop by this fraction from the peak (still also min steps).")
    ap.add_argument("--adaptive_k_probe_steps", type=int, default=200,
                    help="Per-board visits after entering a k stage used to estimate the peak loss EMA for that board.")
    ap.add_argument("--adaptive_k_min_steps", type=int, default=400,
                    help="Minimum visits each unfinished board must receive at the current k before it can be marked ready.")
    ap.add_argument("--adaptive_k_ema_beta", type=float, default=0.98,
                    help="EMA beta for tracking loss when using adaptive_k.")
    ap.add_argument("--adaptive_k_max_steps_per_k", type=int, default=8000,
                    help="If a board stays at the current k for this many visits, require a relaxed target check instead of unconditional ready. 0 disables.")
    ap.add_argument("--adaptive_k_max_visits_target_slack", type=float, default=1.08,
                    help="When max_steps_per_k is hit, only mark k-ready if EMA <= target * slack.")
    ap.add_argument("--adaptive_k_ready_ratio", type=float, default=0.91,
                    help="Promote to the next k once this fraction of unfinished boards are either ready or deferred.")
    ap.add_argument("--adaptive_k_outlier_release_factor", type=float, default=3.0,
                    help="After this multiple of max_steps_per_k visits, a long-hold board is deferred so it no longer blocks promotion. <=1 disables.")

    ap.add_argument("--max_main_steps", type=int, default=0,
                    help="Optional safety cap for the main phase. <=0 disables.")
    ap.add_argument("--board_full_ema_beta", type=float, default=0.90,
                    help="EMA beta for each board after it reaches full-board training.")
    ap.add_argument("--board_full_probe_visits", type=int, default=3,
                    help="How many full-board visits to use for estimating the peak EMA of that board.")
    ap.add_argument("--board_full_drop_ratio", type=float, default=0.20,
                    help="Require a board's full-board EMA loss to drop by this ratio from its peak before graduation.")
    ap.add_argument("--board_full_min_visits", type=int, default=5,
                    help="Minimum full-board visits before a board can graduate.")
    ap.add_argument("--board_full_plateau_patience", type=int, default=3,
                    help="Recent full-board EMA history length used to judge whether the board has stabilized.")
    ap.add_argument("--board_full_plateau_rel_change", type=float, default=0.02,
                    help="Maximum relative EMA change allowed across the recent full-board window to call it stable.")
    ap.add_argument("--board_full_max_visits", type=int, default=0,
                    help="Optional safety graduation cap after reaching full-board. <=0 disables.")

    args = ap.parse_args()
    args.sequence_policy = normalize_sequence_policy(args.sequence_policy)

    device_t = torch.device(args.device)
    if device_t.type != "cuda":
        raise ValueError(f"scripts/step3_train_masked.py is CUDA-only; got --device {args.device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is False.")

    env_source = vars(args)
    if bool(args.resume) and bool(args.resume_restore_config):
        if not args.checkpoint_path:
            raise ValueError("--resume requires --checkpoint_path.")
        resume_meta = torch.load(args.checkpoint_path, map_location="cpu")
        if not isinstance(resume_meta, dict):
            raise ValueError(f"Resume checkpoint must contain a dict payload: {args.checkpoint_path}")
        saved_sequence_policy = resume_meta.get("sequence_policy")
        if saved_sequence_policy is not None:
            args.sequence_policy = normalize_sequence_policy(saved_sequence_policy)
            print(
                f"[resume] expert snapping/validation restoring sequence_policy="
                f"{args.sequence_policy!r} from checkpoint"
            )

        saved_env = resume_meta.get("env_config")
        if isinstance(saved_env, dict):
            env_source = saved_env
            print("[resume] expert snapping/validation restoring PlacementEnv config from checkpoint")
        else:
            print("[resume] checkpoint has no env_config; expert snapping/validation uses current CLI config")

    env_kwargs = build_placement_env_kwargs(env_source)
    validate_placement_env_kwargs(env_kwargs)
    print(f"[env] expert snapping/validation uses PlacementEnv kwargs: {env_kwargs}")

    paths = sorted(glob.glob(args.train_glob))
    if not paths:
        raise SystemExit("No training files matched.")

    train_tasks = []
    for p in paths:
        acts = _expert_actions_from_json(
            p,
            snap_radius=args.expert_snap_radius,
            global_fallback=bool(args.expert_snap_global_fallback),
            env_kwargs=env_kwargs,
            device=device_t,
            sequence_policy=args.sequence_policy,
        )
        _validate_expert(
            p,
            acts,
            env_kwargs=env_kwargs,
            device=device_t,
            sequence_policy=args.sequence_policy,
        )
        train_tasks.append({"path": p, "expert_actions": acts})

    train(
        train_tasks,
        steps=args.steps,
        warmup=args.warmup,
        kmax_final=args.kmax_final,
        adaptive_k=bool(args.adaptive_k),
        fixed_k_steps=int(args.fixed_k_steps),
        curriculum_window_mode=args.curriculum_window_mode,
        adaptive_k_drop_ratio=float(args.adaptive_k_drop_ratio),
        adaptive_k_drop_ratio_first=float(args.adaptive_k_drop_ratio_first),
        adaptive_k_probe_steps=int(args.adaptive_k_probe_steps),
        adaptive_k_min_steps=int(args.adaptive_k_min_steps),
        adaptive_k_ema_beta=float(args.adaptive_k_ema_beta),
        adaptive_k_max_steps_per_k=int(args.adaptive_k_max_steps_per_k),
        adaptive_k_max_visits_target_slack=float(args.adaptive_k_max_visits_target_slack),
        adaptive_k_ready_ratio=float(args.adaptive_k_ready_ratio),
        adaptive_k_outlier_release_factor=float(args.adaptive_k_outlier_release_factor),
        max_main_steps=int(args.max_main_steps),
        board_full_ema_beta=float(args.board_full_ema_beta),
        board_full_probe_visits=int(args.board_full_probe_visits),
        board_full_drop_ratio=float(args.board_full_drop_ratio),
        board_full_min_visits=int(args.board_full_min_visits),
        board_full_plateau_patience=int(args.board_full_plateau_patience),
        board_full_plateau_rel_change=float(args.board_full_plateau_rel_change),
        board_full_max_visits=int(args.board_full_max_visits),
        lr=args.lr,
        seed=int(args.seed),
        save_path=args.save_path,
        device=args.device,
        max_tokens=int(args.max_tokens),
        sequence_policy=args.sequence_policy,
        model_d_model=int(args.model_d_model),
        model_nhead=int(args.model_nhead),
        model_num_layers=int(args.model_num_layers),
        model_dropout=float(args.model_dropout),
        batch_size=int(args.batch_size),
        min_spacing_mm=args.min_spacing_mm,
        env_alignment_bonus=args.env_alignment_bonus,
        env_edge_bonus=args.env_edge_bonus,
        env_edge_eps_mm=args.env_edge_eps_mm,
        reward_non_interface_edge_penalty=args.reward_non_interface_edge_penalty,
        reward_non_interface_edge_margin_mm=args.reward_non_interface_edge_margin_mm,
        reward_density_penalty=args.reward_density_penalty,
        reward_density_radius_mm=args.reward_density_radius_mm,
        reward_interior_penalty=args.reward_interior_penalty,
        reward_interior_margin_ratio=args.reward_interior_margin_ratio,

        teacher_tau=args.teacher_tau,
        teacher_lambda_region_prior=args.teacher_lambda_region_prior,
        teacher_lambda_prior_region_heatmap=args.teacher_lambda_prior_region_heatmap,
        teacher_lambda_conn_prior=args.teacher_lambda_conn_prior,
        teacher_lambda_anchor_prior=args.teacher_lambda_anchor_prior,
        teacher_lambda_module_prior=args.teacher_lambda_module_prior,
        teacher_lambda_spacing_prior=args.teacher_lambda_spacing_prior,
        teacher_lambda_edge_prior=args.teacher_lambda_edge_prior,
        teacher_metric_weight=args.teacher_metric_weight,
        objective_hpwl_weight=args.objective_hpwl_weight,
        objective_w_hpwl_weight=args.objective_w_hpwl_weight,
        objective_nslw_weight=args.objective_nslw_weight,
        objective_region_weight=args.objective_region_weight,
        objective_module_region_weight=args.objective_module_region_weight,
        objective_module_floorplan_weight=args.objective_module_floorplan_weight,
        module_region_bias=args.module_region_bias,
        module_region_margin_mm=args.module_region_margin_mm,
        module_floorplan_separation_mm=args.module_floorplan_separation_mm,
        module_floorplan_overlap_scale=args.module_floorplan_overlap_scale,
        module_floorplan_channel_scale=args.module_floorplan_channel_scale,
        module_floorplan_compact_scale=args.module_floorplan_compact_scale,
        module_floorplan_region_scale=args.module_floorplan_region_scale,
        objective_conn_weight=args.objective_conn_weight,
        objective_align_weight=args.objective_align_weight,
        objective_group_weight=args.objective_group_weight,
        objective_anchor_weight=args.objective_anchor_weight,
        objective_boundary_group_weight=args.objective_boundary_group_weight,
        objective_pitch_weight=args.objective_pitch_weight,
        objective_orientation_weight=args.objective_orientation_weight,
        objective_edge_clearance_weight=args.objective_edge_clearance_weight,
        objective_interior_weight=args.objective_interior_weight,
        objective_density_weight=args.objective_density_weight,
        objective_soft_spacing_weight=args.objective_soft_spacing_weight,
        objective_neatness_weight=args.objective_neatness_weight,
        edge_band_ratio=args.edge_band_ratio,
        edge_band_center_ratio=args.edge_band_center_ratio,
        soft_spacing_same_group_extra_mm=args.soft_spacing_same_group_extra_mm,
        soft_spacing_cross_group_extra_mm=args.soft_spacing_cross_group_extra_mm,
        soft_spacing_large_extra_mm=args.soft_spacing_large_extra_mm,
        same_group_density_scale=args.same_group_density_scale,
        critical_neighbor_density_scale=args.critical_neighbor_density_scale,
        anchor_group_density_scale=args.anchor_group_density_scale,
        large_pair_density_scale=args.large_pair_density_scale,
        region_prior_enabled=bool(args.region_prior_enabled),
        region_grid_x=args.region_grid_x,
        region_grid_y=args.region_grid_y,
        region_heatmap_sigma_cells=args.region_heatmap_sigma_cells,
        legacy_region_zone_edge_ratio=args.legacy_region_zone_edge_ratio,
        legacy_region_zone_core_ratio=args.legacy_region_zone_core_ratio,
        region_heatmap_action_prior_weight=args.region_heatmap_action_prior_weight,
        region_aux_heatmap_weight=args.region_aux_heatmap_weight,
        region_aux_prior_consistency_weight=args.region_aux_prior_consistency_weight,
        region_aux_semantic_weight=args.region_aux_semantic_weight,
        region_aux_side_weight=args.region_aux_side_weight,
        region_aux_subzone_weight=args.region_aux_subzone_weight,
        region_aux_pairwise_weight=args.region_aux_pairwise_weight,
        teacher_topk=args.teacher_topk,
        teacher_objective_delta_max=args.teacher_objective_delta_max,
        teacher_gate_rollout=bool(args.teacher_gate_rollout),

        expert_mix_start=args.expert_mix_start,
        expert_mix_end=args.expert_mix_end,
        expert_mix_anneal_steps=args.expert_mix_anneal_steps,

        geometry_xy_weight=args.geometry_xy_weight,
        geometry_rot_weight=args.geometry_rot_weight,
        geometry_align_offset_weight=args.geometry_align_offset_weight,
        geometry_boundary_axis_weight=args.geometry_boundary_axis_weight,
        geometry_module_bbox_weight=args.geometry_module_bbox_weight,

        replay_finetune=bool(args.replay_finetune),
        replay_iters=args.replay_iters,
        replay_rollouts_per_iter=args.replay_rollouts_per_iter,
        replay_update_steps=args.replay_update_steps,
        replay_batch_size=args.replay_batch_size,
        replay_capacity=args.replay_capacity,
        replay_alpha=args.replay_alpha,
        replay_temp=args.replay_temp,
        replay_action_ce_coef=args.replay_action_ce_coef,
        replay_policy_gradient_coef=args.replay_policy_gradient_coef,
        replay_entropy_coef=args.replay_entropy_coef,
        replay_rollout_temperature=args.replay_rollout_temperature,
        replay_pg_policy_mode=args.replay_pg_policy_mode,
        replay_baseline_beta=args.replay_baseline_beta,
        replay_advantage_clip=args.replay_advantage_clip,
        replay_k=args.replay_k,
        visdom_enabled=bool(args.visdom_enabled),
        visdom_server=args.visdom_server,
        visdom_port=int(args.visdom_port),
        visdom_env=args.visdom_env,
        visdom_prefix=args.visdom_prefix,
        visdom_log_interval=int(args.visdom_log_interval),
        checkpoint_path=args.checkpoint_path,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        resume=bool(args.resume),
        resume_restore_config=bool(args.resume_restore_config),
    )


if __name__ == "__main__":
    main()
