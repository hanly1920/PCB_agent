from __future__ import annotations
import math
import os
import random
import signal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from .dataset import task_from_json
from .utils import coarse_type_from_fine
from .env import PlacementEnv
from .model import MaskedPolicy, ModelConfig
from .region_prior import (
    REGION_TYPE_NAMES,
    SEMANTIC_CLASS_NAMES,
    SIDE_PREFERENCE_NAMES,
    SUBZONE_NAMES,
    PAIRWISE_RELATION_NAMES,
    RegionPriorConfig,
    action_region_prior_from_predictions,
    build_action_region_indices,
    load_region_targets_for_task,
    semantic_review_weight,
)
from .env_cuda import objective_delta_mask_cuda, build_teacher_distribution_cuda


def _flatten_action(rix: int, ix: int, iy: int, w: int, h: int, R: int) -> int:
    return rix * (w * h) + ix * h + iy


def _unflatten_action(a: int, w: int, h: int, R: int) -> Tuple[int, int, int]:
    wh = w * h
    rix = a // wh
    rem = a % wh
    ix = rem // h
    iy = rem % h
    return int(rix), int(ix), int(iy)


def schedule_kmax(step: int, total_steps: int, hold_steps: int, kmax_final: int, power: float = 1.8) -> int:
    """
    Full-masking friendly k schedule:
    - Hold k=1 for `hold_steps`
    - Then cosine ramp to kmax_final by `total_steps`
    power>1 makes early growth slower (lower variance), later faster.
    """
    if kmax_final <= 1:
        return 1
    hold_steps = max(0, int(hold_steps))
    total_steps = max(1, int(total_steps))

    if step <= hold_steps:
        return 1

    denom = max(1, total_steps - hold_steps)
    u = min(1.0, float(step - hold_steps) / float(denom))           # 0..1
    s = 0.5 * (1.0 - math.cos(math.pi * u))                         # smooth 0..1
    s = s ** float(power)                                           # slower early, faster late

    k = 1.0 + (float(kmax_final) - 1.0) * s
    return max(1, min(kmax_final, int(round(k))))


def schedule_mix(step: int, total_steps: int, hold_steps: int, start: float, end: float, power: float = 1.6) -> float:
    """
    Cosine + hold schedule for alpha_expert:
    - Hold at `start` for `hold_steps`
    - Then cosine decay to `end` by `total_steps`
    """
    total_steps = max(1, int(total_steps))
    hold_steps = max(0, int(hold_steps))

    if total_steps <= hold_steps + 1:
        return float(end)
    if step <= hold_steps:
        return float(start)

    denom = max(1, total_steps - hold_steps)
    u = min(1.0, float(step - hold_steps) / float(denom))
    s = 0.5 * (1.0 - math.cos(math.pi * u))
    s = s ** float(power)
    return float(start + (end - start) * s)


_TYPES = ["interface", "mechanical", "chip", "capacitor", "resistor", "misc"]


def _comp_nets(env: PlacementEnv, ref: str) -> set[str]:
    c = env.comp_by_ref[ref]
    nets = {str(n) for (n, _) in c.pads if n and str(n).upper() not in ("", "GND", "GROUND")}
    return nets



def _action_features(env: PlacementEnv) -> np.ndarray:
    """Per-action feature matrix aligned with env.observe() flatten order [R,w,h].

    Features: [x_norm, y_norm, rot_sin, rot_cos]  (float32), shape [A,4]
    """
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw = max(1e-6, (xmax - xmin))
    bh = max(1e-6, (ymax - ymin))
    grid = float(env.task.grid_mm)
    w, h = env.grid_shape()
    R = len(env.rotations)

    # normalized centers
    xs = xmin + (np.arange(w, dtype=np.float32) + 0.5) * grid
    ys = ymin + (np.arange(h, dtype=np.float32) + 0.5) * grid
    xn = (xs - xmin) / bw
    yn = (ys - ymin) / bh
    Xn, Yn = np.meshgrid(xn, yn, indexing="ij")  # [w,h]

    feats = np.zeros((R, w, h, 4), dtype=np.float32)
    for ri, rot in enumerate(env.rotations):
        rr = float(rot) * math.pi / 180.0
        rs = math.sin(rr)
        rc = math.cos(rr)
        feats[ri, :, :, 0] = Xn
        feats[ri, :, :, 1] = Yn
        feats[ri, :, :, 2] = rs
        feats[ri, :, :, 3] = rc

    return feats.reshape(-1, 4)

def build_context_tokens(env: PlacementEnv, ref: str, max_tokens: int = 128) -> np.ndarray:
    """Build a sequence of tokens for the baseline transformer policy.

    Tokens = (selected placed components) + (current component).
    For each placed token, include its normalized position/rotation and whether it shares nets with current.
    """
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw = max(1e-6, (xmax - xmin))
    bh = max(1e-6, (ymax - ymin))

    cur = env.comp_by_ref[ref]
    cur_nets = _comp_nets(env, ref)

    keep = build_context_ref_order(env, ref, max_tokens=max_tokens)

    tokens: List[np.ndarray] = []
    placed_ratio = env.t / max(1, len(env.sequence))

    def make_token_for_placed(r: str) -> np.ndarray:
        c = env.comp_by_ref[r]
        one = np.zeros((len(_TYPES),), dtype=np.float32)
        t0 = coarse_type_from_fine(c.type)
        t = t0 if t0 in _TYPES else "misc"
        one[_TYPES.index(t)] = 1.0

        x, y, rot = env.placed[r]
        xn = (x - xmin) / bw
        yn = (y - ymin) / bh
        rot_rad = math.radians(rot)
        rs = math.sin(rot_rad)
        rc = math.cos(rot_rad)

        w, h = c.size_mm
        wn = w / bw
        hn = h / bh

        sh = float(len(cur_nets.intersection(_comp_nets(env, r))))
        shn = min(1.0, sh / 8.0)

        # [type(6), size(2), pos(2), placed(1), rot_sin_cos(2), shared(1), placed_ratio(1), is_current(1)] = 16
        return np.concatenate(
            [one, np.array([wn, hn, xn, yn, 1.0, rs, rc, shn, placed_ratio, 0.0], dtype=np.float32)], axis=0
        )

    def make_token_for_current() -> np.ndarray:
        one = np.zeros((len(_TYPES),), dtype=np.float32)
        t0 = coarse_type_from_fine(cur.type)
        t = t0 if t0 in _TYPES else "misc"
        one[_TYPES.index(t)] = 1.0
        w, h = cur.size_mm
        wn = w / bw
        hn = h / bh
        return np.concatenate(
            [one, np.array([wn, hn, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, placed_ratio, 1.0], dtype=np.float32)],
            axis=0,
        )

    for r in keep[::-1]:  # older -> newer
        tokens.append(make_token_for_placed(r))
    tokens.append(make_token_for_current())
    return np.stack(tokens, axis=0).astype(np.float32)



def build_context_ref_order(env: PlacementEnv, ref: str, max_tokens: int = 128) -> List[str]:
    cur_nets = _comp_nets(env, ref)
    placed_refs = list(env.placed_order)
    shared = []
    other = []
    for r in reversed(placed_refs):
        sh = len(cur_nets.intersection(_comp_nets(env, r)))
        (shared if sh > 0 else other).append((r, sh))
    ordered = [r for (r, _) in sorted(shared, key=lambda x: (-x[1],))] + [r for (r, _) in other]
    keep = ordered[:max(0, max_tokens - 1)]
    return list(reversed(keep))

def _best_legal_action(mask_flat: np.ndarray, bias_flat: np.ndarray) -> Optional[int]:
    idx = np.where(mask_flat > 0.5)[0]
    if idx.size == 0:
        return None
    best = idx[int(np.argmax(bias_flat[idx]))]
    return int(best)


@dataclass
class TeacherConfig:
    """Teacher distribution q(a) built from the unified objective delta ΔJ(a)."""
    tau: float = 0.5
    lambda_region_prior: float = 0.12
    metric_weight: float = 0.15
    topk: int = 256
    objective_delta_max: Optional[float] = None
    gate_rollout: bool = True


@dataclass
class GeometryDistillConfig:
    xy_weight: float = 0.05
    rot_weight: float = 0.02
    align_offset_weight: float = 0.03
    boundary_axis_weight: float = 0.03


def _teacher_candidate_mask(
    mask_flat: np.ndarray,
    objective_flat: np.ndarray,
    energy: np.ndarray,
    topk: int,
    objective_delta_max: Optional[float],
) -> np.ndarray:
    """Return a boolean candidate mask over actions."""
    legal = mask_flat > 0.5
    cand = legal.copy()

    if objective_delta_max is not None:
        cand = cand & (objective_flat <= float(objective_delta_max))
        if not np.any(cand):
            cand = legal.copy()

    if topk is not None and int(topk) > 0:
        idx = np.where(cand)[0]
        if idx.size > int(topk):
            sel = idx[np.argsort(energy[idx])[: int(topk)]]
            new_cand = np.zeros_like(cand, dtype=bool)
            new_cand[sel] = True
            cand = new_cand
    return cand


def build_teacher_distribution(
    mask_flat: np.ndarray,
    objective_flat: np.ndarray,
    cfg: TeacherConfig,
    device: torch.device,
    region_flat: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Build q(a) from the unified objective delta and optional learned region prior."""
    energy = objective_flat.astype(np.float32).copy()
    if region_flat is not None:
        energy = energy - float(cfg.lambda_region_prior) * np.asarray(region_flat, dtype=np.float32)
    cand = _teacher_candidate_mask(mask_flat, objective_flat, energy, cfg.topk, cfg.objective_delta_max)

    tlog = -energy / max(1e-6, float(cfg.tau))
    tlog2 = np.full_like(tlog, -1e9, dtype=np.float32)
    tlog2[cand] = tlog[cand].astype(np.float32)

    qt = torch.from_numpy(tlog2).to(device)
    q = torch.softmax(qt, dim=-1)
    return q, cand


def _policy_outputs_with_region(
    model: MaskedPolicy,
    env: PlacementEnv,
    ref: str,
    tokens_t: torch.Tensor,
    feat_t: torch.Tensor,
    region_cfg: RegionPriorConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray], Optional[torch.Tensor], Optional[List[str]]]:
    enc = model.encode(tokens_t)
    q_last = model.q_proj(enc[:, -1, :])
    logits = model.action_logits_from_query(q_last, feat_t).squeeze(0)

    if not bool(region_cfg.enabled):
        return logits, None, None, None, None, None, None, enc, None

    region_type_logits = model.region_type_logits(q_last).squeeze(0)
    semantic_class_logits = model.semantic_class_logits(q_last).squeeze(0)
    side_preference_logits = model.side_preference_logits(q_last).squeeze(0)
    subzone_logits = model.subzone_logits(q_last).squeeze(0)
    pairwise_logits = None
    token_refs = build_context_ref_order(env, ref, max_tokens=tokens_t.shape[1])
    if enc.shape[1] > 1:
        peer_enc = enc[:, :-1, :].mean(dim=1)
        pairwise_logits = model.pairwise_relation_logits(q_last, peer_enc).squeeze(0)
    region_idx = build_action_region_indices(
        env,
        ref,
        grid_x=int(region_cfg.grid_x),
        grid_y=int(region_cfg.grid_y),
        zone_edge_ratio=float(region_cfg.zone_edge_ratio),
        zone_core_ratio=float(region_cfg.zone_core_ratio),
    )
    region_prior = action_region_prior_from_predictions(
        region_type_logits,
        region_type_idx_flat=region_idx,
        zone_prior_weight=float(region_cfg.zone_prior_weight),
    )
    return logits, region_type_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior, enc, token_refs



def _ref_expected_norm_from_probs(
    probs: torch.Tensor,
    feat_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_exp = (probs * feat_t[:, 0]).sum()
    y_exp = (probs * feat_t[:, 1]).sum()
    sin_exp = (probs * feat_t[:, 2]).sum()
    cos_exp = (probs * feat_t[:, 3]).sum()
    return x_exp, y_exp, sin_exp, cos_exp


def _align_axis_target(
    env: PlacementEnv,
    ref: str,
) -> Optional[Tuple[str, float]]:
    group = getattr(env, "_align_groups", {}).get(ref)
    if not group:
        return None
    peers = [pref for pref in getattr(env, "placed_order", []) if pref != ref and getattr(env, "_align_groups", {}).get(pref) == group]
    if not peers:
        return None
    xs = [float(env.placed[pref][0]) for pref in peers]
    ys = [float(env.placed[pref][1]) for pref in peers]
    if len(xs) >= 2 and len(ys) >= 2:
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        axis = "x" if spread_x <= spread_y else "y"
    else:
        axis = "x" if len(xs) >= len(ys) else "y"
    target = float(sum(xs) / max(1, len(xs))) if axis == "x" else float(sum(ys) / max(1, len(ys)))
    return axis, target


def _boundary_axis_target(
    env: PlacementEnv,
    ref: str,
) -> Optional[Tuple[str, str]]:
    group = getattr(env, "_same_side_groups", {}).get(ref)
    if not group:
        return None
    peers = [pref for pref in getattr(env, "placed_order", []) if pref != ref and getattr(env, "_same_side_groups", {}).get(pref) == group]
    if not peers:
        return None
    side = getattr(env, "_side_preferences", {}).get(ref, "free")
    if side not in {"edge_left", "edge_right", "edge_top", "edge_bottom"}:
        side = getattr(env, "_region_targets", {}).get(ref, "free")
    if side in {"edge_left", "edge_right"}:
        return "y", side
    if side in {"edge_top", "edge_bottom"}:
        return "x", side
    return None


def _geometry_distill_loss(
    *,
    env: PlacementEnv,
    ref: str,
    logits: torch.Tensor,
    feat_t: torch.Tensor,
    expert_action: Tuple[int, int, int],
    geom_cfg: GeometryDistillConfig,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    x_exp, y_exp, sin_exp, cos_exp = _ref_expected_norm_from_probs(probs, feat_t)

    w, h = env.grid_shape()
    R = len(env.rotations)
    gt_a = _flatten_action(int(expert_action[0]), int(expert_action[1]), int(expert_action[2]), w, h, R)
    tgt = feat_t[gt_a]

    loss = torch.tensor(0.0, device=logits.device)

    pred_xy = torch.stack([x_exp, y_exp], dim=0)
    tgt_xy = tgt[:2]
    loss = loss + float(geom_cfg.xy_weight) * F.smooth_l1_loss(pred_xy, tgt_xy, reduction="sum")

    pred_rot = torch.stack([sin_exp, cos_exp], dim=0)
    tgt_rot = tgt[2:4]
    loss = loss + float(geom_cfg.rot_weight) * F.smooth_l1_loss(pred_rot, tgt_rot, reduction="sum")

    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw = max(1e-6, float(xmax - xmin))
    bh = max(1e-6, float(ymax - ymin))

    align_target = _align_axis_target(env, ref)
    if align_target is not None:
        axis, axis_value = align_target
        if axis == "x":
            pred_off = x_exp - (float(axis_value) - float(xmin)) / bw
            tgt_off = tgt[0] - (float(axis_value) - float(xmin)) / bw
        else:
            pred_off = y_exp - (float(axis_value) - float(ymin)) / bh
            tgt_off = tgt[1] - (float(axis_value) - float(ymin)) / bh
        loss = loss + float(geom_cfg.align_offset_weight) * F.smooth_l1_loss(
            pred_off.view(1), tgt_off.view(1), reduction="sum"
        )

    boundary_target = _boundary_axis_target(env, ref)
    if boundary_target is not None:
        axis, _side = boundary_target
        pred_axis = x_exp if axis == "x" else y_exp
        tgt_axis = tgt[0] if axis == "x" else tgt[1]
        loss = loss + float(geom_cfg.boundary_axis_weight) * F.smooth_l1_loss(
            pred_axis.view(1), tgt_axis.view(1), reduction="sum"
        )

    return loss

def _region_aux_losses(
    *,
    ref: str,
    task_path: str,
    env: PlacementEnv,
    model: MaskedPolicy,
    region_cfg: RegionPriorConfig,
    region_type_logits: Optional[torch.Tensor],
    semantic_class_logits: Optional[torch.Tensor],
    side_preference_logits: Optional[torch.Tensor],
    subzone_logits: Optional[torch.Tensor],
    pairwise_logits: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    zero = torch.tensor(0.0, device=device)
    if (not bool(region_cfg.enabled)) or region_type_logits is None or semantic_class_logits is None:
        return zero, zero, zero, zero, zero, 1.0
    targets = load_region_targets_for_task(
        task_path,
        grid_x=int(region_cfg.grid_x),
        grid_y=int(region_cfg.grid_y),
        sigma_cells=float(region_cfg.heatmap_sigma_cells),
        zone_edge_ratio=float(region_cfg.zone_edge_ratio),
        zone_core_ratio=float(region_cfg.zone_core_ratio),
    )
    tgt = targets.get(ref)
    if not tgt:
        return zero, zero, zero, zero, zero, 1.0

    review_weight = float(tgt.get("review_weight", 1.0))
    region_t = torch.tensor([int(tgt["region_type"])], device=device)
    semantic_t = torch.tensor([int(tgt["semantic_class"])], device=device)
    side_t = torch.tensor([int(tgt.get("side_preference", 0))], device=device)
    subzone_t = torch.tensor([int(tgt.get("subzone", 0))], device=device)
    loss_region_type = F.cross_entropy(region_type_logits.view(1, -1), region_t)
    loss_semantic = F.cross_entropy(semantic_class_logits.view(1, -1), semantic_t)
    loss_side = zero if side_preference_logits is None else F.cross_entropy(side_preference_logits.view(1, -1), side_t)
    loss_subzone = zero if subzone_logits is None else F.cross_entropy(subzone_logits.view(1, -1), subzone_t)

    relation_target = torch.zeros((1, len(PAIRWISE_RELATION_NAMES)), device=device)
    placed = list(env.placed_order)
    if placed:
        relation_target[0, 0] = 1.0 if any(env._functional_group.get(p, 'misc') == env._functional_group.get(ref, 'misc') for p in placed) else 0.0
        relation_target[0, 1] = 1.0 if any((env._anchor_refs.get(ref) == p) or (env._anchor_refs.get(p) == ref) for p in placed) else 0.0
        relation_target[0, 2] = 1.0 if any((p in env._critical_neighbors.get(ref, ())) or (ref in env._critical_neighbors.get(p, ())) for p in placed) else 0.0
    loss_pairwise = zero if pairwise_logits is None else F.binary_cross_entropy_with_logits(pairwise_logits.view(1, -1), relation_target)
    return loss_semantic, loss_region_type, loss_side, loss_subzone, loss_pairwise, review_weight



def rollout_suffix_loss(
    model: MaskedPolicy,
    env: PlacementEnv,
    expert_actions: List[Tuple[int, int, int]],
    task_path: str,
    k: int,
    alpha_expert: float,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    device: torch.device,
    *,
    do_backward: bool = False,
    grad_weight: float = 1.0,
) -> torch.Tensor:
    """Explicit suffix-masked training, but supervision is fused:
    - expert label (CE) mixed with
    - teacher distribution q(a) from the same unified objective delta used by env reward, using KL(q||pi) i.e. cross-entropy(q, pi)

    When do_backward=True:
      - we do backward() **per suffix step** to avoid keeping all suffix graphs until the end (prevents OOM)
      - gradient is scaled by 1/(T-prefix_len) so it matches the original mean-loss scale (approximately; if env terminates early, scale differs slightly)
      - the returned tensor is a detached scalar for logging
    """
    T = len(env.sequence)
    prefix_len = max(0, T - k)
    w, h = env.grid_shape()
    R = len(env.rotations)
    A = R * w * h

    # number of intended suffix steps (used for scaling so gradients match mean(losses))
    suffix_steps_target = max(1, T - prefix_len)
    step_scale = float(grad_weight) / float(suffix_steps_target)

    # Precompute action features once (constant within an episode)
    feat = _action_features(env)
    feat_t = torch.from_numpy(feat).to(device)

    # (Optional) collect losses only when we are NOT doing per-step backward
    losses: List[torch.Tensor] = []

    # For logging (no graph retention)
    loss_sum = 0.0
    loss_cnt = 0

    # Teacher-force prefix with expert, but fail fast if illegal
    for t in range(prefix_len):
        obs, _, _, info = env.step(expert_actions[t])
        if info.get("illegal"):
            raise ValueError(f"Illegal expert action at t={t} ref={obs.get('ref')} info={info}")

    for t in range(prefix_len, T):
        if env.done():
            break

        ref = env.current_ref()
        obs = env.observe()
        mask = obs["action_mask"].reshape(-1).astype(np.float32)  # [A]
        bias = obs["action_bias"].reshape(-1).astype(np.float32)  # [A]

        if float(mask.max()) < 0.5:
            env.terminated = True
            break

        obj_maps = objective_delta_mask_cuda(env, ref, device)
        objective_delta_t = obj_maps["total"].reshape(-1)

        tokens = build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device)

        logits, region_type_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior, _enc_ctx, _token_refs = _policy_outputs_with_region(
            model, env, ref, tokens_t, feat_t, region_cfg
        )
        m = torch.from_numpy(mask).to(device)
        logits = logits.masked_fill(m < 0.5, -1e9)

        loss_semantic, loss_region_type, loss_side, loss_subzone, loss_pairwise, review_weight = _region_aux_losses(
            ref=ref,
            task_path=task_path,
            env=env,
            model=model,
            region_cfg=region_cfg,
            region_type_logits=region_type_logits,
            semantic_class_logits=semantic_class_logits,
            side_preference_logits=side_preference_logits,
            subzone_logits=subzone_logits,
            pairwise_logits=pairwise_logits,
            device=device,
        )
        review_weight = float(max(0.0, min(1.0, review_weight)))
        region_prior_t = None if region_prior is None else (torch.from_numpy(np.asarray(region_prior, dtype=np.float32)).to(device) * review_weight)

        mask_t = torch.from_numpy(mask).to(device)
        q, cand = build_teacher_distribution_cuda(mask_t, objective_delta_t, teacher, device=device, region_flat=region_prior_t)
        logp = torch.log_softmax(logits, dim=-1)
        loss_teacher = -(q * logp).sum()
        probs = torch.softmax(logits, dim=-1)
        loss_metric = (probs * objective_delta_t).sum()

        # Expert CE (repair if needed)
        gt = expert_actions[t]
        gt_a = _flatten_action(gt[0], gt[1], gt[2], w, h, R)
        if gt_a < 0 or gt_a >= A or mask[gt_a] < 0.5:
            repaired = _best_legal_action(mask, bias)
            if repaired is None:
                env.terminated = True
                break
            gt_a = repaired
        loss_expert = F.cross_entropy(logits.view(1, -1), torch.tensor([gt_a], device=device))
        loss_geo = _geometry_distill_loss(
            env=env,
            ref=ref,
            logits=logits,
            feat_t=feat_t,
            expert_action=gt,
            geom_cfg=geom_cfg,
        )

        aexp = float(alpha_expert)
        aexp_eff = aexp + (1.0 - aexp) * (1.0 - review_weight)
        loss = (
            aexp_eff * loss_expert
            + (1.0 - aexp_eff) * loss_teacher
            + review_weight * float(teacher.metric_weight) * loss_metric
            + aexp_eff * loss_geo
            + review_weight * float(region_cfg.aux_heatmap_weight) * loss_semantic
            + review_weight * float(region_cfg.aux_zone_weight) * loss_region_type
            + review_weight * float(region_cfg.aux_side_weight) * loss_side
            + review_weight * float(region_cfg.aux_subzone_weight) * loss_subzone
            + review_weight * float(region_cfg.aux_pairwise_weight) * loss_pairwise
        )

        # === OOM FIX: backward per step (optional) ===
        if do_backward:
            (loss * step_scale).backward()
        else:
            losses.append(loss)

        # logging stats (no graph)
        loss_sum += float(loss.detach().item())
        loss_cnt += 1

        if teacher.gate_rollout:
            logits_roll = logits.masked_fill(~cand, -1e9)
        else:
            logits_roll = logits

        a = int(torch.argmax(logits_roll).item())
        _, _, _, info = env.step(_unflatten_action(a, w, h, R))
        if info.get("illegal"):
            break

    if do_backward:
        # return a detached scalar for logging
        if loss_cnt <= 0:
            return torch.tensor(0.0, device=device)
        return torch.tensor(loss_sum / float(loss_cnt), device=device)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()



# -------------------------
# Weighted Replay Fine-tune
# -------------------------
class WeightedReplayBuffer:
    def __init__(self, capacity: int = 2000, alpha: float = 0.7):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.items: List[Dict[str, Any]] = []
        self.priorities: List[float] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: Dict[str, Any], priority: float) -> None:
        p = float(max(1e-8, priority))
        if len(self.items) >= self.capacity:
            self.items.pop(0)
            self.priorities.pop(0)
        self.items.append(item)
        self.priorities.append(p)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.items:
            return []
        ps = np.array(self.priorities, dtype=np.float64) ** self.alpha
        ps = ps / ps.sum()
        idx = np.random.choice(len(self.items), size=min(batch_size, len(self.items)), replace=False, p=ps)
        return [self.items[int(i)] for i in idx]


def rollout_episode(
    model: MaskedPolicy,
    task_path: str,
    device: torch.device,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Roll out a full episode using current policy (with optional HPWL-guided gating)."""
    task = task_from_json(task_path)
    env = PlacementEnv(task, **(env_kwargs or {}))
    w, h = env.grid_shape()
    R = len(env.rotations)
    actions_flat: List[int] = []
    rewards: List[float] = []

    while not env.done():
        ref = env.current_ref()
        obs = env.observe()
        mask = obs["action_mask"].reshape(-1).astype(np.float32)
        bias = obs["action_bias"].reshape(-1).astype(np.float32)

        if float(mask.max()) < 0.5:
            env.terminated = True
            break

        obj_maps = objective_delta_mask_cuda(env, ref, device)
        objective_delta_t = obj_maps["total"].reshape(-1)
        tokens = build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device)

        with torch.no_grad():
            feat = _action_features(env)
            feat_t = torch.from_numpy(feat).to(device)
            logits, _region_type_logits, _semantic_class_logits, _side_logits, _subzone_logits, _pairwise_logits, region_prior, _enc_ctx, _token_refs = _policy_outputs_with_region(
                model, env, ref, tokens_t, feat_t, region_cfg
            )
            m = torch.from_numpy(mask).to(device)
            logits = logits.masked_fill(m < 0.5, -1e9)

            if teacher.gate_rollout:
                mask_t = torch.from_numpy(mask).to(device)
                region_prior_t = None if region_prior is None else torch.from_numpy(np.asarray(region_prior, dtype=np.float32)).to(device)
                _q, cand = build_teacher_distribution_cuda(mask_t, objective_delta_t, teacher, device=device, region_flat=region_prior_t)
                logits = logits.masked_fill(~cand, -1e9)

            a = int(torch.argmax(logits).item())

        _obs2, r, _done, info = env.step(_unflatten_action(a, w, h, R))
        if info.get("illegal"):
            break
        actions_flat.append(int(a))
        rewards.append(float(r))

    total_return = float(sum(rewards))
    final_obj = float(env.prev_obj)
    score = -final_obj  # higher is better (min objective)
    return {
        "task_path": task_path,
        "actions_flat": actions_flat,
        "total_return": total_return,
        "final_obj": final_obj,
        "score": score,
        "terminated": bool(env.terminated),
        "steps": int(len(actions_flat)),
    }


def replay_teacher_loss_on_episode(
    model: MaskedPolicy,
    ep: Dict[str, Any],
    k: int,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    expert_actions_by_path: Dict[str, List[Tuple[int, int, int]]],
    device: torch.device,
    action_ce_coef: float = 0.1,
    env_kwargs: Optional[Dict[str, Any]] = None,
    *,
    do_backward: bool = False,
    grad_weight: float = 1.0,
) -> torch.Tensor:
    """Reconstruct states by replaying the episode's actions, then train on teacher q(a) (plus small CE on the episode action).

    This keeps the same suffix-masked training form, but uses self-generated trajectories to provide on-policy states.

    When do_backward=True:
      - backward() is called **per suffix step** (prevents OOM)
      - gradient is scaled by 1/(#suffix steps in this episode) to match mean-loss scale
      - returns a detached scalar for logging
    """
    task = task_from_json(ep["task_path"])
    env = PlacementEnv(task, **(env_kwargs or {}))
    w, h = env.grid_shape()
    R = len(env.rotations)
    T = len(env.sequence)
    prefix_len = max(0, T - k)

    actions_flat: List[int] = [int(a) for a in ep.get("actions_flat", [])]
    expert_actions = expert_actions_by_path.get(ep["task_path"], [])
    if not actions_flat:
        return torch.tensor(0.0, device=device, requires_grad=(not do_backward))

    # Precompute action features once (constant within an episode)
    feat = _action_features(env)
    feat_t = torch.from_numpy(feat).to(device)

    # suffix steps that actually exist in this episode (for scaling)
    max_suffix = max(0, min(T, len(actions_flat)) - prefix_len)
    suffix_steps_target = max(1, int(max_suffix))
    step_scale = float(grad_weight) / float(suffix_steps_target)

    # (Optional) collect losses only when we are NOT doing per-step backward
    losses: List[torch.Tensor] = []

    loss_sum = 0.0
    loss_cnt = 0

    # Teacher-force prefix with episode actions (truncate if episode is shorter)
    for t in range(min(prefix_len, len(actions_flat))):
        a = int(actions_flat[t])
        _obs2, _r, _done, info = env.step(_unflatten_action(a, w, h, R))
        if info.get("illegal") or env.done():
            return torch.tensor(0.0, device=device, requires_grad=(not do_backward))

    # Train on suffix steps that exist in the episode
    for t in range(prefix_len, min(T, len(actions_flat))):
        if env.done():
            break
        ref = env.current_ref()
        obs = env.observe()
        mask = obs["action_mask"].reshape(-1).astype(np.float32)
        bias = obs["action_bias"].reshape(-1).astype(np.float32)
        if float(mask.max()) < 0.5:
            env.terminated = True
            break

        obj_maps = objective_delta_mask_cuda(env, ref, device)
        objective_delta_t = obj_maps["total"].reshape(-1)
        tokens = build_context_tokens(env, ref)[None, :, :]
        tokens_t = torch.from_numpy(tokens).to(device)

        logits, region_type_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior, _enc_ctx, _token_refs = _policy_outputs_with_region(
            model, env, ref, tokens_t, feat_t, region_cfg
        )
        m = torch.from_numpy(mask).to(device)
        logits = logits.masked_fill(m < 0.5, -1e9)

        loss_semantic, loss_region_type, loss_side, loss_subzone, loss_pairwise, review_weight = _region_aux_losses(
            ref=ref,
            task_path=ep["task_path"],
            env=env,
            model=model,
            region_cfg=region_cfg,
            region_type_logits=region_type_logits,
            semantic_class_logits=semantic_class_logits,
            side_preference_logits=side_preference_logits,
            subzone_logits=subzone_logits,
            pairwise_logits=pairwise_logits,
            device=device,
        )
        review_weight = float(max(0.0, min(1.0, review_weight)))
        region_prior_t = None if region_prior is None else (torch.from_numpy(np.asarray(region_prior, dtype=np.float32)).to(device) * review_weight)

        mask_t = torch.from_numpy(mask).to(device)
        q, _cand = build_teacher_distribution_cuda(mask_t, objective_delta_t, teacher, device=device, region_flat=region_prior_t)
        logp = torch.log_softmax(logits, dim=-1)
        loss_teacher = -(q * logp).sum()
        probs = torch.softmax(logits, dim=-1)
        loss_metric = (probs * objective_delta_t).sum()

        # small anchoring to the episode action (only if legal)
        a_flat = int(actions_flat[t])
        if 0 <= a_flat < logits.shape[0] and mask[a_flat] > 0.5:
            loss_act = F.cross_entropy(logits.view(1, -1), torch.tensor([a_flat], device=device))
        else:
            loss_act = torch.tensor(0.0, device=device)

        if t < len(expert_actions):
            loss_geo = _geometry_distill_loss(
                env=env,
                ref=ref,
                logits=logits,
                feat_t=feat_t,
                expert_action=expert_actions[t],
                geom_cfg=geom_cfg,
            )
        else:
            loss_geo = torch.tensor(0.0, device=device)

        loss = (
            review_weight * loss_teacher
            + float(action_ce_coef) * loss_act
            + review_weight * float(teacher.metric_weight) * loss_metric
            + loss_geo
            + review_weight * float(region_cfg.aux_heatmap_weight) * loss_semantic
            + review_weight * float(region_cfg.aux_zone_weight) * loss_region_type
            + review_weight * float(region_cfg.aux_side_weight) * loss_side
            + review_weight * float(region_cfg.aux_subzone_weight) * loss_subzone
            + review_weight * float(region_cfg.aux_pairwise_weight) * loss_pairwise
        )

        # === OOM FIX: backward per step (optional) ===
        if do_backward:
            (loss * step_scale).backward()
        else:
            losses.append(loss)

        loss_sum += float(loss.detach().item())
        loss_cnt += 1

        # step using stored action
        _obs2, _r, _done, info = env.step(_unflatten_action(a_flat, w, h, R))
        if info.get("illegal"):
            break

    if do_backward:
        if loss_cnt <= 0:
            return torch.tensor(0.0, device=device)
        return torch.tensor(loss_sum / float(loss_cnt), device=device)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()




@dataclass
class BoardCurriculumState:
    index: int
    path: str
    kmax: int
    k_stage: int = 1
    k_stage_visits: int = 0
    k_stage_loss_ema: Optional[float] = None
    k_stage_peak_ema: Optional[float] = None
    k_stage_target_ema: Optional[float] = None
    k_stage_target_locked: bool = False
    k_stage_probe_left: int = 0
    k_stage_ready: bool = False
    k_stage_deferred: bool = False
    k_stage_defer_reason: Optional[str] = None
    graduated: bool = False
    full_visits: int = 0
    full_loss_ema: Optional[float] = None
    full_peak_ema: Optional[float] = None
    full_target_ema: Optional[float] = None
    full_probe_left: int = 0
    recent_full_ema: List[float] = field(default_factory=list)
    graduate_step: Optional[int] = None
    graduate_reason: Optional[str] = None


def _reset_board_k_stage(st: BoardCurriculumState, k_stage: int) -> None:
    st.k_stage = int(max(1, k_stage))
    st.k_stage_visits = 0
    st.k_stage_loss_ema = None
    st.k_stage_peak_ema = None
    st.k_stage_target_ema = None
    st.k_stage_target_locked = False
    st.k_stage_probe_left = 0
    st.k_stage_ready = False
    st.k_stage_deferred = False
    st.k_stage_defer_reason = None


def _ensure_board_k_stage(st: BoardCurriculumState, k_stage: int) -> None:
    if int(st.k_stage) != int(k_stage):
        _reset_board_k_stage(st, k_stage)


def _update_board_k_stage_state(
    st: BoardCurriculumState,
    *,
    k_stage: int,
    loss_val: float,
    ema_beta: float,
    probe_steps: int,
    drop_ratio: float,
    min_visits: int,
    max_visits: int = 0,
    max_visits_target_slack: float = 1.08,
    outlier_release_factor: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    _ensure_board_k_stage(st, k_stage)

    beta = min(max(float(ema_beta), 0.0), 0.999)
    if st.k_stage_loss_ema is None:
        st.k_stage_loss_ema = float(loss_val)
    else:
        st.k_stage_loss_ema = beta * float(st.k_stage_loss_ema) + (1.0 - beta) * float(loss_val)

    st.k_stage_visits += 1
    ema = float(st.k_stage_loss_ema)

    if st.k_stage_target_ema is not None:
        st.k_stage_target_locked = True

    if st.k_stage_probe_left <= 0 and (not st.k_stage_target_locked) and st.k_stage_target_ema is None:
        st.k_stage_probe_left = max(1, int(probe_steps))
        st.k_stage_peak_ema = ema

    if st.k_stage_probe_left > 0 and (not st.k_stage_target_locked):
        st.k_stage_peak_ema = ema if st.k_stage_peak_ema is None else max(float(st.k_stage_peak_ema), ema)
        st.k_stage_probe_left -= 1
        if st.k_stage_probe_left == 0 and st.k_stage_target_ema is None:
            st.k_stage_target_ema = float(st.k_stage_peak_ema) * (1.0 - _clip01(drop_ratio))
            st.k_stage_target_locked = True

    enough_visits = st.k_stage_visits >= max(1, int(min_visits))
    target_ready = st.k_stage_target_ema is not None and ema <= float(st.k_stage_target_ema)
    max_visits_hit = int(max_visits) > 0 and st.k_stage_visits >= int(max_visits)
    slack = max(1.0, float(max_visits_target_slack))
    forced_target_ready = (
        max_visits_hit
        and enough_visits
        and st.k_stage_target_ema is not None
        and ema <= float(st.k_stage_target_ema) * slack
    )
    outlier_release_hit = (
        int(max_visits) > 0
        and float(outlier_release_factor) > 1.0
        and st.k_stage_visits >= int(math.ceil(float(outlier_release_factor) * int(max_visits)))
    )

    if enough_visits and target_ready:
        st.k_stage_ready = True
        st.k_stage_deferred = False
        st.k_stage_defer_reason = None
        return True, 'stable'

    if forced_target_ready:
        st.k_stage_ready = True
        st.k_stage_deferred = False
        st.k_stage_defer_reason = None
        return True, 'max_visits_target'

    if outlier_release_hit and not st.k_stage_ready and not st.k_stage_deferred:
        st.k_stage_deferred = True
        st.k_stage_defer_reason = 'outlier_release'
        return False, 'outlier_deferred'

    if max_visits_hit:
        st.k_stage_ready = False
        return False, 'max_visits_hold'

    st.k_stage_ready = False
    return False, None


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 0.999)


def _recent_rel_change_small(vals: List[float], eps: float) -> bool:
    if len(vals) <= 1:
        return False
    eps = max(0.0, float(eps))
    rels: List[float] = []
    for prev, curr in zip(vals[:-1], vals[1:]):
        denom = max(abs(float(prev)), 1e-6)
        rels.append(abs(float(curr) - float(prev)) / denom)
    return bool(rels) and max(rels) <= eps


def _update_board_full_state(
    st: BoardCurriculumState,
    loss_val: float,
    main_step: int,
    ema_beta: float,
    probe_visits: int,
    drop_ratio: float,
    min_visits: int,
    plateau_patience: int,
    plateau_rel_change: float,
    max_visits: int = 0,
) -> Tuple[bool, Optional[str]]:
    beta = min(max(float(ema_beta), 0.0), 0.999)
    if st.full_loss_ema is None:
        st.full_loss_ema = float(loss_val)
    else:
        st.full_loss_ema = beta * float(st.full_loss_ema) + (1.0 - beta) * float(loss_val)

    st.full_visits += 1
    ema = float(st.full_loss_ema)

    if st.full_probe_left <= 0 and st.full_target_ema is None:
        st.full_probe_left = max(1, int(probe_visits))
        st.full_peak_ema = ema

    if st.full_probe_left > 0:
        st.full_peak_ema = ema if st.full_peak_ema is None else max(float(st.full_peak_ema), ema)
        st.full_probe_left -= 1
        if st.full_probe_left == 0:
            st.full_target_ema = float(st.full_peak_ema) * (1.0 - _clip01(drop_ratio))

    hist_keep = max(2, int(plateau_patience) + 1)
    st.recent_full_ema.append(ema)
    if len(st.recent_full_ema) > hist_keep:
        st.recent_full_ema = st.recent_full_ema[-hist_keep:]

    enough_visits = st.full_visits >= max(int(min_visits), int(probe_visits) + int(plateau_patience))
    target_ready = st.full_target_ema is not None and ema <= float(st.full_target_ema)
    plateau_ready = len(st.recent_full_ema) >= hist_keep and _recent_rel_change_small(
        st.recent_full_ema[-hist_keep:], float(plateau_rel_change)
    )

    if enough_visits and target_ready and plateau_ready:
        st.graduated = True
        st.graduate_step = int(main_step)
        st.graduate_reason = 'stable'
        return True, st.graduate_reason

    if int(max_visits) > 0 and st.full_visits >= int(max_visits):
        st.graduated = True
        st.graduate_step = int(main_step)
        st.graduate_reason = 'max_visits'
        return True, st.graduate_reason

    return False, None



def _board_state_to_dict(st: BoardCurriculumState) -> Dict[str, Any]:
    return {
        'index': int(st.index),
        'path': st.path,
        'kmax': int(st.kmax),
        'k_stage': int(st.k_stage),
        'k_stage_visits': int(st.k_stage_visits),
        'k_stage_loss_ema': None if st.k_stage_loss_ema is None else float(st.k_stage_loss_ema),
        'k_stage_peak_ema': None if st.k_stage_peak_ema is None else float(st.k_stage_peak_ema),
        'k_stage_target_ema': None if st.k_stage_target_ema is None else float(st.k_stage_target_ema),
        'k_stage_target_locked': bool(st.k_stage_target_locked),
        'k_stage_probe_left': int(st.k_stage_probe_left),
        'k_stage_ready': bool(st.k_stage_ready),
        'k_stage_deferred': bool(st.k_stage_deferred),
        'k_stage_defer_reason': st.k_stage_defer_reason,
        'graduated': bool(st.graduated),
        'full_visits': int(st.full_visits),
        'full_loss_ema': None if st.full_loss_ema is None else float(st.full_loss_ema),
        'full_peak_ema': None if st.full_peak_ema is None else float(st.full_peak_ema),
        'full_target_ema': None if st.full_target_ema is None else float(st.full_target_ema),
        'full_probe_left': int(st.full_probe_left),
        'recent_full_ema': [float(v) for v in st.recent_full_ema],
        'graduate_step': None if st.graduate_step is None else int(st.graduate_step),
        'graduate_reason': st.graduate_reason,
    }


def _board_state_from_dict(d: Dict[str, Any]) -> BoardCurriculumState:
    st = BoardCurriculumState(
        index=int(d['index']),
        path=str(d['path']),
        kmax=int(d['kmax']),
    )
    st.k_stage = int(d.get('k_stage', 1))
    st.k_stage_visits = int(d.get('k_stage_visits', 0))
    st.k_stage_loss_ema = None if d.get('k_stage_loss_ema') is None else float(d.get('k_stage_loss_ema'))
    st.k_stage_peak_ema = None if d.get('k_stage_peak_ema') is None else float(d.get('k_stage_peak_ema'))
    st.k_stage_target_ema = None if d.get('k_stage_target_ema') is None else float(d.get('k_stage_target_ema'))
    st.k_stage_target_locked = bool(d.get('k_stage_target_locked', st.k_stage_target_ema is not None))
    st.k_stage_probe_left = int(d.get('k_stage_probe_left', 0))
    st.k_stage_ready = bool(d.get('k_stage_ready', False))
    st.k_stage_deferred = bool(d.get('k_stage_deferred', False))
    st.k_stage_defer_reason = d.get('k_stage_defer_reason')
    st.graduated = bool(d.get('graduated', False))
    st.full_visits = int(d.get('full_visits', 0))
    st.full_loss_ema = None if d.get('full_loss_ema') is None else float(d.get('full_loss_ema'))
    st.full_peak_ema = None if d.get('full_peak_ema') is None else float(d.get('full_peak_ema'))
    st.full_target_ema = None if d.get('full_target_ema') is None else float(d.get('full_target_ema'))
    st.full_probe_left = int(d.get('full_probe_left', 0))
    st.recent_full_ema = [float(v) for v in d.get('recent_full_ema', [])]
    st.graduate_step = None if d.get('graduate_step') is None else int(d.get('graduate_step'))
    st.graduate_reason = d.get('graduate_reason')
    return st


def _make_step_checkpoint_path(base_path: str, phase: str, step: int) -> str:
    p = Path(base_path)
    suffix = ''.join(p.suffixes)
    stem = p.name[:-len(suffix)] if suffix else p.name
    filename = f'{stem}.{phase}.step{int(step):07d}.pt'
    return str(p.with_name(filename))


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_name(path_obj.name + '.tmp')
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path_obj))


def _cleanup_old_checkpoints(base_path: str, keep_last: int) -> List[str]:
    if int(keep_last) <= 0:
        return []
    p = Path(base_path)
    stem = p.stem
    candidates = sorted(
        p.parent.glob(f'{stem}.*.step*.pt'),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for old in candidates[int(keep_last):]:
        try:
            old.unlink()
            removed.append(str(old))
        except FileNotFoundError:
            pass
    return removed


def _save_checkpoint(
    checkpoint_path: str,
    payload: Dict[str, Any],
    phase: str,
    step: int,
    keep_last: int,
) -> Tuple[str, str, List[str]]:
    latest_path = str(Path(checkpoint_path))
    step_path = _make_step_checkpoint_path(latest_path, phase=phase, step=int(step))
    _atomic_torch_save(payload, step_path)
    _atomic_torch_save(payload, latest_path)
    removed = _cleanup_old_checkpoints(latest_path, int(keep_last))
    return latest_path, step_path, removed


def _serialize_replay_buffer(rb: Optional[WeightedReplayBuffer]) -> Optional[Dict[str, Any]]:
    if rb is None:
        return None
    return {
        'capacity': int(rb.capacity),
        'alpha': float(rb.alpha),
        'items': list(rb.items),
        'priorities': [float(v) for v in rb.priorities],
    }


def _restore_replay_buffer(data: Optional[Dict[str, Any]]) -> Optional[WeightedReplayBuffer]:
    if not data:
        return None
    rb = WeightedReplayBuffer(capacity=int(data.get('capacity', 2000)), alpha=float(data.get('alpha', 0.7)))
    rb.items = list(data.get('items', []))
    rb.priorities = [float(v) for v in data.get('priorities', [])]
    return rb


def _build_checkpoint_payload(
    *,
    model: MaskedPolicy,
    opt: AdamW,
    obs_dim: int,
    max_tokens: int,
    board_states: List[BoardCurriculumState],
    kmax_final: int,
    warmup: int,
    steps: int,
    max_main_steps: int,
    adaptive_k: bool,
    adaptive_k_drop_ratio: float,
    adaptive_k_drop_ratio_first: float,
    adaptive_k_probe_steps: int,
    adaptive_k_min_steps: int,
    adaptive_k_ema_beta: float,
    adaptive_k_max_steps_per_k: int,
    board_full_ema_beta: float,
    board_full_probe_visits: int,
    board_full_drop_ratio: float,
    board_full_min_visits: int,
    board_full_plateau_patience: int,
    board_full_plateau_rel_change: float,
    board_full_max_visits: int,
    env_alignment_bonus: float,
    env_edge_bonus: float,
    env_edge_eps_mm: float,
    reward_non_interface_edge_penalty: float,
    reward_non_interface_edge_margin_mm: float,
    reward_density_penalty: float,
    reward_density_radius_mm: float,
    reward_interior_penalty: float,
    reward_interior_margin_ratio: float,
    objective_nslw_weight: float,
    objective_region_weight: float,
    objective_conn_weight: float,
    objective_align_weight: float,
    objective_group_weight: float,
    objective_anchor_weight: float,
    objective_boundary_group_weight: float,
    objective_pitch_weight: float,
    objective_orientation_weight: float,
    objective_edge_clearance_weight: float,
    objective_interior_weight: float,
    objective_density_weight: float,
    objective_soft_spacing_weight: float,
    objective_neatness_weight: float,
    edge_band_ratio: float,
    edge_band_center_ratio: float,
    soft_spacing_same_group_extra_mm: float,
    soft_spacing_cross_group_extra_mm: float,
    soft_spacing_large_extra_mm: float,
    same_group_density_scale: float,
    critical_neighbor_density_scale: float,
    anchor_group_density_scale: float,
    large_pair_density_scale: float,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    expert_mix_start: float,
    expert_mix_end: float,
    expert_mix_anneal_steps: int,
    replay_finetune: bool,
    replay_iters: int,
    replay_rollouts_per_iter: int,
    replay_update_steps: int,
    replay_batch_size: int,
    replay_capacity: int,
    replay_alpha: float,
    replay_temp: float,
    replay_action_ce_coef: float,
    replay_k: Optional[int],
    main_step: int,
    rr_ptr: int,
    step_cap_hit: bool,
    k_curr: int,
    loss_ema: Optional[float],
    k_steps: int,
    probe_left: int,
    peak_ema: Optional[float],
    target_ema: Optional[float],
    phase: str,
    replay_iter_done: int,
    replay_buffer: Optional[WeightedReplayBuffer],
    replay_best_score: Optional[float],
) -> Dict[str, Any]:
    return {
        'format_version': 3,
        'phase': phase,
        'replay_iter_done': int(replay_iter_done),
        'replay_best_score': None if replay_best_score is None else float(replay_best_score),
        'replay_buffer': _serialize_replay_buffer(replay_buffer),
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'obs_dim': obs_dim,
        'model_cfg': ModelConfig().__dict__,
        'action_feat_dim': 4,
        'region_grid_shape': [int(region_cfg.grid_x), int(region_cfg.grid_y)],
        'num_region_types': int(len(REGION_TYPE_NAMES)),
        'num_semantic_classes': int(len(SEMANTIC_CLASS_NAMES)),
        'num_side_preferences': int(len(SIDE_PREFERENCE_NAMES)),
        'num_subzones': int(len(SUBZONE_NAMES)),
        'num_pairwise_relations': int(len(PAIRWISE_RELATION_NAMES)),
        'max_tokens': int(max_tokens),
        'k_schedule': {
            'kmax_final': int(kmax_final),
            'warmup': int(warmup),
            'steps': int(steps),
            'max_main_steps': int(max_main_steps),
            'adaptive': bool(adaptive_k),
            'adaptive_drop_ratio': float(adaptive_k_drop_ratio),
            'adaptive_drop_ratio_first': float(adaptive_k_drop_ratio_first),
            'adaptive_probe_steps': int(adaptive_k_probe_steps),
            'adaptive_min_steps': int(adaptive_k_min_steps),
            'adaptive_ema_beta': float(adaptive_k_ema_beta),
            'adaptive_max_steps_per_k': int(adaptive_k_max_steps_per_k),
        },
        'board_graduation': {
            'ema_beta': float(board_full_ema_beta),
            'probe_visits': int(board_full_probe_visits),
            'drop_ratio': float(board_full_drop_ratio),
            'min_visits': int(board_full_min_visits),
            'plateau_patience': int(board_full_plateau_patience),
            'plateau_rel_change': float(board_full_plateau_rel_change),
            'max_visits': int(board_full_max_visits),
        },
        'env_config': {
            'alignment_bonus': float(env_alignment_bonus),
            'edge_bonus': float(env_edge_bonus),
            'edge_eps_mm': float(env_edge_eps_mm),
            'non_interface_edge_penalty': float(reward_non_interface_edge_penalty),
            'non_interface_edge_margin_mm': float(reward_non_interface_edge_margin_mm),
            'density_penalty': float(reward_density_penalty),
            'density_radius_mm': float(reward_density_radius_mm),
            'interior_penalty': float(reward_interior_penalty),
            'interior_margin_ratio': float(reward_interior_margin_ratio),
            'nslw_weight': float(objective_nslw_weight),
            'region_weight': float(objective_region_weight),
            'conn_weight': float(objective_conn_weight),
            'objective_align_weight': float(objective_align_weight),
            'group_weight': float(objective_group_weight),
            'anchor_weight': float(objective_anchor_weight),
            'boundary_group_weight': float(objective_boundary_group_weight),
            'pitch_weight': float(objective_pitch_weight),
            'orientation_weight': float(objective_orientation_weight),
            'objective_edge_clearance_weight': float(objective_edge_clearance_weight),
            'objective_interior_weight': float(objective_interior_weight),
            'objective_density_weight': float(objective_density_weight),
            'objective_soft_spacing_weight': float(objective_soft_spacing_weight),
            'objective_neatness_weight': float(objective_neatness_weight),
            'edge_band_ratio': float(edge_band_ratio),
            'edge_band_center_ratio': float(edge_band_center_ratio),
            'soft_spacing_same_group_extra_mm': float(soft_spacing_same_group_extra_mm),
            'soft_spacing_cross_group_extra_mm': float(soft_spacing_cross_group_extra_mm),
            'soft_spacing_large_extra_mm': float(soft_spacing_large_extra_mm),
            'same_group_density_scale': float(same_group_density_scale),
            'critical_neighbor_density_scale': float(critical_neighbor_density_scale),
            'anchor_group_density_scale': float(anchor_group_density_scale),
            'large_pair_density_scale': float(large_pair_density_scale),
        },
        'board_states': [_board_state_to_dict(st) for st in board_states],
        'main_training': {
            'main_steps_executed': int(main_step),
            'step_cap_hit': bool(step_cap_hit),
            'all_boards_graduated': bool(all(st.graduated for st in board_states)),
            'rr_ptr': int(rr_ptr),
            'k_curr': int(k_curr),
            'loss_ema': None if loss_ema is None else float(loss_ema),
            'k_steps': int(k_steps),
            'probe_left': int(probe_left),
            'peak_ema': None if peak_ema is None else float(peak_ema),
            'target_ema': None if target_ema is None else float(target_ema),
            'board_k_tracking': {
                'per_board': True,
                'all_growth_boards_ready_required': True,
            },
        },
        'teacher': {
            'tau': float(teacher.tau),
            'lambda_region_prior': float(teacher.lambda_region_prior),
            'metric_weight': float(teacher.metric_weight),
            'topk': int(teacher.topk),
            'objective_delta_max': None if teacher.objective_delta_max is None else float(teacher.objective_delta_max),
            'gate_rollout': bool(teacher.gate_rollout),
        },
        'region_prior': {
            'enabled': bool(region_cfg.enabled),
            'grid_x': int(region_cfg.grid_x),
            'grid_y': int(region_cfg.grid_y),
            'heatmap_sigma_cells': float(region_cfg.heatmap_sigma_cells),
            'zone_edge_ratio': float(region_cfg.zone_edge_ratio),
            'zone_core_ratio': float(region_cfg.zone_core_ratio),
            'zone_prior_weight': float(region_cfg.zone_prior_weight),
            'aux_heatmap_weight': float(region_cfg.aux_heatmap_weight),
            'aux_zone_weight': float(region_cfg.aux_zone_weight),
            'aux_side_weight': float(region_cfg.aux_side_weight),
            'aux_subzone_weight': float(region_cfg.aux_subzone_weight),
            'aux_pairwise_weight': float(region_cfg.aux_pairwise_weight),
            'region_type_names': list(REGION_TYPE_NAMES),
            'semantic_class_names': list(SEMANTIC_CLASS_NAMES),
        },
        'geometry_distill': {
            'xy_weight': float(geom_cfg.xy_weight),
            'rot_weight': float(geom_cfg.rot_weight),
            'align_offset_weight': float(geom_cfg.align_offset_weight),
            'boundary_axis_weight': float(geom_cfg.boundary_axis_weight),
        },
        'expert_mix': {
            'start': float(expert_mix_start),
            'end': float(expert_mix_end),
            'anneal_steps': int(expert_mix_anneal_steps),
        },
        'replay': {
            'enabled': bool(replay_finetune),
            'iters': int(replay_iters),
            'rollouts_per_iter': int(replay_rollouts_per_iter),
            'update_steps': int(replay_update_steps),
            'batch_size': int(replay_batch_size),
            'capacity': int(replay_capacity),
            'alpha': float(replay_alpha),
            'temp': float(replay_temp),
            'action_ce_coef': float(replay_action_ce_coef),
            'k': None if replay_k is None else int(replay_k),
        },
    }


def train(
    train_tasks: List[Dict[str, Any]],
    steps: int = 0,
    warmup: int = 1500,
    kmax_final: int = -1,
    # k curriculum
    adaptive_k: bool = False,
    adaptive_k_drop_ratio: float = 0.7,
    adaptive_k_drop_ratio_first: float = 0.4,
    adaptive_k_probe_steps: int = 200,
    adaptive_k_min_steps: int = 400,
    adaptive_k_ema_beta: float = 0.98,
    adaptive_k_max_steps_per_k: int = 2500,
    adaptive_k_max_visits_target_slack: float = 1.08,
    adaptive_k_ready_ratio: float = 0.91,
    adaptive_k_outlier_release_factor: float = 3.0,
    max_main_steps: int = 0,
    # per-board graduation after full-board training
    board_full_ema_beta: float = 0.90,
    board_full_probe_visits: int = 3,
    board_full_drop_ratio: float = 0.20,
    board_full_min_visits: int = 5,
    board_full_plateau_patience: int = 3,
    board_full_plateau_rel_change: float = 0.02,
    board_full_max_visits: int = 0,
    lr: float = 3e-4,
    seed: int = 7,
    save_path: str = 'model.pt',
    device: str = 'cuda',
    max_tokens: int = 128,
    env_alignment_bonus: float = 0.05,
    env_edge_bonus: float = 0.15,
    env_edge_eps_mm: float = 1.5,
    reward_non_interface_edge_penalty: float = 10.0,
    reward_non_interface_edge_margin_mm: float = 2.5,
    reward_density_penalty: float = 3.0,
    reward_density_radius_mm: float = 4.0,
    reward_interior_penalty: float = 1.0,
    reward_interior_margin_ratio: float = 0.18,

    # 1) soft teacher q(a)
    teacher_tau: float = 0.5,
    teacher_lambda_region_prior: float = 0.12,
    teacher_metric_weight: float = 0.15,
    objective_nslw_weight: float = 0.2,
    objective_region_weight: float = 0.55,
    objective_conn_weight: float = 0.50,
    objective_align_weight: float = 0.28,
    objective_group_weight: float = 0.12,
    objective_anchor_weight: float = 0.18,
    objective_boundary_group_weight: float = 0.22,
    objective_pitch_weight: float = 0.22,
    objective_orientation_weight: float = 0.14,
    objective_edge_clearance_weight: float = 0.40,
    objective_interior_weight: float = 0.30,
    objective_density_weight: float = 0.45,
    objective_soft_spacing_weight: float = 0.32,
    objective_neatness_weight: float = 0.12,
    edge_band_ratio: float = 0.12,
    edge_band_center_ratio: float = 0.55,
    soft_spacing_same_group_extra_mm: float = 0.6,
    soft_spacing_cross_group_extra_mm: float = 1.4,
    soft_spacing_large_extra_mm: float = 0.7,
    same_group_density_scale: float = 0.40,
    critical_neighbor_density_scale: float = 0.25,
    anchor_group_density_scale: float = 0.50,
    large_pair_density_scale: float = 1.20,

    region_prior_enabled: bool = True,
    region_grid_x: int = 6,
    region_grid_y: int = 6,
    region_heatmap_sigma_cells: float = 0.85,
    region_zone_edge_ratio: float = 0.12,
    region_zone_core_ratio: float = 0.28,
    region_zone_prior_weight: float = 0.35,
    region_aux_heatmap_weight: float = 0.30,
    region_aux_zone_weight: float = 0.10,

    geometry_xy_weight: float = 0.05,
    geometry_rot_weight: float = 0.02,
    geometry_align_offset_weight: float = 0.03,
    geometry_boundary_axis_weight: float = 0.03,

    # 3) Top-K / threshold teacher (objective-guided)
    teacher_topk: int = 256,
    teacher_objective_delta_max: Optional[float] = None,
    teacher_gate_rollout: bool = True,

    # 2) expert mix annealing
    expert_mix_start: float = 1.0,
    expert_mix_end: float = 0.40,
    expert_mix_anneal_steps: int = 20000,

    # 4) weighted replay fine-tune
    replay_finetune: bool = True,
    replay_iters: int = 20,
    replay_rollouts_per_iter: int = 16,
    replay_update_steps: int = 32,
    replay_batch_size: int = 4,
    replay_capacity: int = 2000,
    replay_alpha: float = 0.7,
    replay_temp: float = 1.0,
    replay_action_ce_coef: float = 0.1,
    replay_k: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every_steps: int = 0,
    keep_last_checkpoints: int = 0,
    resume: bool = False,
):
    """Training with fused masked learning.

    Main phase:
      - Keep a global k curriculum (adaptive or scheduled).
      - Every board uses k_eff = min(global_k, board_kmax).
      - Once a board reaches full-board training (k_eff == board_kmax), it no longer waits for a fixed
        number of visits. Instead, it maintains its own EMA loss, target EMA, and plateau history.
      - When that board's full-board loss both drops enough and plateaus, the board graduates and is
        removed from the main training pool.

    Replay phase:
      - After all boards graduate (or a safety step cap is hit), run the existing weighted replay fine-tune.
    """
    if not train_tasks:
        raise ValueError('train_tasks is empty.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_t = torch.device(device)

    Tmax = max(len(it['expert_actions']) for it in train_tasks)
    if int(kmax_final) <= 0 or int(kmax_final) < Tmax:
        if int(kmax_final) > 0 and int(kmax_final) < Tmax:
            print(
                f'[warn] kmax_final={int(kmax_final)} < Tmax={Tmax}; '
                'raise to Tmax so every board can reach full-board and graduate.'
            )
        kmax_final = Tmax
        print(f'[info] full masking enabled: kmax_final={kmax_final} (Tmax={Tmax})')

    obs_dim = 16
    region_cfg = RegionPriorConfig(
        enabled=bool(region_prior_enabled),
        grid_x=int(region_grid_x),
        grid_y=int(region_grid_y),
        heatmap_sigma_cells=float(region_heatmap_sigma_cells),
        zone_edge_ratio=float(region_zone_edge_ratio),
        zone_core_ratio=float(region_zone_core_ratio),
        zone_prior_weight=float(region_zone_prior_weight),
        aux_heatmap_weight=float(region_aux_heatmap_weight),
        aux_zone_weight=float(region_aux_zone_weight),
        aux_side_weight=0.12,
        aux_subzone_weight=0.10,
        aux_pairwise_weight=0.10,
    )
    model = MaskedPolicy(
        obs_dim=obs_dim,
        cfg=ModelConfig(),
        region_grid_shape=(int(region_cfg.grid_x), int(region_cfg.grid_y)),
        num_region_types=int(len(REGION_TYPE_NAMES)),
        num_semantic_classes=int(len(SEMANTIC_CLASS_NAMES)),
        num_side_preferences=int(len(SIDE_PREFERENCE_NAMES)),
        num_subzones=int(len(SUBZONE_NAMES)),
        num_pairwise_relations=int(len(PAIRWISE_RELATION_NAMES)),
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr)

    teacher = TeacherConfig(
        tau=float(teacher_tau),
        lambda_region_prior=float(teacher_lambda_region_prior),
        metric_weight=float(teacher_metric_weight),
        topk=int(teacher_topk),
        objective_delta_max=None if teacher_objective_delta_max is None else float(teacher_objective_delta_max),
        gate_rollout=bool(teacher_gate_rollout),
    )
    geom_cfg = GeometryDistillConfig(
        xy_weight=float(geometry_xy_weight),
        rot_weight=float(geometry_rot_weight),
        align_offset_weight=float(geometry_align_offset_weight),
        boundary_axis_weight=float(geometry_boundary_axis_weight),
    )
    expert_actions_by_path: Dict[str, List[Tuple[int, int, int]]] = {
        str(it['path']): list(it['expert_actions']) for it in train_tasks
    }

    env_kwargs: Dict[str, Any] = {
        'alignment_bonus': float(env_alignment_bonus),
        'edge_bonus': float(env_edge_bonus),
        'edge_eps_mm': float(env_edge_eps_mm),
        'non_interface_edge_penalty': float(reward_non_interface_edge_penalty),
        'non_interface_edge_margin_mm': float(reward_non_interface_edge_margin_mm),
        'density_penalty': float(reward_density_penalty),
        'density_radius_mm': float(reward_density_radius_mm),
        'interior_penalty': float(reward_interior_penalty),
        'interior_margin_ratio': float(reward_interior_margin_ratio),
        'nslw_weight': float(objective_nslw_weight),
        'region_weight': float(objective_region_weight),
        'conn_weight': float(objective_conn_weight),
        'objective_align_weight': float(objective_align_weight),
        'group_weight': float(objective_group_weight),
        'anchor_weight': float(objective_anchor_weight),
        'boundary_group_weight': float(objective_boundary_group_weight),
        'pitch_weight': float(objective_pitch_weight),
        'orientation_weight': float(objective_orientation_weight),
    }

    model.train()

    stop_requested: Dict[str, Optional[int]] = {'signal': None}

    def _handle_stop(sig_num, _frame):
        stop_requested['signal'] = int(sig_num)
        print(f'[signal] received signal={int(sig_num)}; checkpoint will be saved at the next safe point.')

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    board_states: List[BoardCurriculumState] = [
        BoardCurriculumState(index=i, path=it['path'], kmax=len(it['expert_actions']))
        for i, it in enumerate(train_tasks)
    ]

    # Adaptive-k controller: global k, but readiness is tracked per board at the current k stage.
    k_curr = 1
    loss_ema: Optional[float] = None
    k_steps = 0
    probe_left = 0
    peak_ema: Optional[float] = None
    target_ema: Optional[float] = None

    schedule_total_steps = int(steps) if int(steps) > 0 else max(
        int(kmax_final) * max(1, int(adaptive_k_min_steps)),
        int(expert_mix_anneal_steps) if int(expert_mix_anneal_steps) > 0 else 0,
        int(warmup) + 1,
    )

    rr_ptr = 0
    main_step = 0
    step_cap_hit = False
    phase = 'main'
    replay_iter_done = 0
    replay_buffer: Optional[WeightedReplayBuffer] = None
    replay_best_score: Optional[float] = None

    if bool(resume):
        if not checkpoint_path:
            raise ValueError('resume=True requires checkpoint_path to be set.')
        ckpt = torch.load(checkpoint_path, map_location=device_t)
        incompatible = model.load_state_dict(ckpt['model_state'], strict=False)
        if getattr(incompatible, 'missing_keys', None) or getattr(incompatible, 'unexpected_keys', None):
            print(f'[resume] state_dict missing_keys={list(incompatible.missing_keys)} unexpected_keys={list(incompatible.unexpected_keys)}')
        opt.load_state_dict(ckpt['optimizer_state'])
        saved_states = [_board_state_from_dict(d) for d in ckpt.get('board_states', [])]
        if len(saved_states) == len(board_states):
            board_states = saved_states
        else:
            print(f'[warn] checkpoint board_states size {len(saved_states)} != current tasks {len(board_states)}; ignore saved board states.')
        mt = ckpt.get('main_training', {})
        rr_ptr = int(mt.get('rr_ptr', 0))
        main_step = int(mt.get('main_steps_executed', 0))
        step_cap_hit = bool(mt.get('step_cap_hit', False))
        k_curr = int(mt.get('k_curr', 1))
        loss_ema = None if mt.get('loss_ema') is None else float(mt.get('loss_ema'))
        k_steps = int(mt.get('k_steps', 0))
        probe_left = int(mt.get('probe_left', 0))
        peak_ema = None if mt.get('peak_ema') is None else float(mt.get('peak_ema'))
        target_ema = None if mt.get('target_ema') is None else float(mt.get('target_ema'))
        phase = str(ckpt.get('phase', 'main'))
        replay_iter_done = int(ckpt.get('replay_iter_done', 0))
        replay_best_score = None if ckpt.get('replay_best_score') is None else float(ckpt.get('replay_best_score'))
        replay_buffer = _restore_replay_buffer(ckpt.get('replay_buffer'))
        print(f'[resume] loaded checkpoint: {checkpoint_path}')
        print(f'[resume] phase={phase} main_step={main_step} replay_iter_done={replay_iter_done}')

    def _checkpoint_now(curr_phase: str, curr_step: int) -> None:
        if not checkpoint_path:
            return
        payload = _build_checkpoint_payload(
            model=model,
            opt=opt,
            obs_dim=obs_dim,
            max_tokens=max_tokens,
            board_states=board_states,
            kmax_final=kmax_final,
            warmup=warmup,
            steps=steps,
            max_main_steps=max_main_steps,
            adaptive_k=adaptive_k,
            adaptive_k_drop_ratio=adaptive_k_drop_ratio,
            adaptive_k_drop_ratio_first=adaptive_k_drop_ratio_first,
            adaptive_k_probe_steps=adaptive_k_probe_steps,
            adaptive_k_min_steps=adaptive_k_min_steps,
            adaptive_k_ema_beta=adaptive_k_ema_beta,
            adaptive_k_max_steps_per_k=adaptive_k_max_steps_per_k,
            board_full_ema_beta=board_full_ema_beta,
            board_full_probe_visits=board_full_probe_visits,
            board_full_drop_ratio=board_full_drop_ratio,
            board_full_min_visits=board_full_min_visits,
            board_full_plateau_patience=board_full_plateau_patience,
            board_full_plateau_rel_change=board_full_plateau_rel_change,
            board_full_max_visits=board_full_max_visits,
            env_alignment_bonus=env_alignment_bonus,
            env_edge_bonus=env_edge_bonus,
            env_edge_eps_mm=env_edge_eps_mm,
            reward_non_interface_edge_penalty=reward_non_interface_edge_penalty,
            reward_non_interface_edge_margin_mm=reward_non_interface_edge_margin_mm,
            reward_density_penalty=reward_density_penalty,
            reward_density_radius_mm=reward_density_radius_mm,
            reward_interior_penalty=reward_interior_penalty,
            reward_interior_margin_ratio=reward_interior_margin_ratio,
            objective_nslw_weight=objective_nslw_weight,
            objective_region_weight=objective_region_weight,
            objective_conn_weight=objective_conn_weight,
            objective_align_weight=objective_align_weight,
            objective_group_weight=objective_group_weight,
            objective_anchor_weight=objective_anchor_weight,
            objective_boundary_group_weight=objective_boundary_group_weight,
            objective_pitch_weight=objective_pitch_weight,
            objective_orientation_weight=objective_orientation_weight,
            objective_edge_clearance_weight=objective_edge_clearance_weight,
            objective_interior_weight=objective_interior_weight,
            objective_density_weight=objective_density_weight,
            objective_soft_spacing_weight=objective_soft_spacing_weight,
            objective_neatness_weight=objective_neatness_weight,
            edge_band_ratio=edge_band_ratio,
            edge_band_center_ratio=edge_band_center_ratio,
            soft_spacing_same_group_extra_mm=soft_spacing_same_group_extra_mm,
            soft_spacing_cross_group_extra_mm=soft_spacing_cross_group_extra_mm,
            soft_spacing_large_extra_mm=soft_spacing_large_extra_mm,
            same_group_density_scale=same_group_density_scale,
            critical_neighbor_density_scale=critical_neighbor_density_scale,
            anchor_group_density_scale=anchor_group_density_scale,
            large_pair_density_scale=large_pair_density_scale,
            teacher=teacher,
            region_cfg=region_cfg,
            geom_cfg=geom_cfg,
            expert_mix_start=expert_mix_start,
            expert_mix_end=expert_mix_end,
            expert_mix_anneal_steps=expert_mix_anneal_steps,
            replay_finetune=replay_finetune,
            replay_iters=replay_iters,
            replay_rollouts_per_iter=replay_rollouts_per_iter,
            replay_update_steps=replay_update_steps,
            replay_batch_size=replay_batch_size,
            replay_capacity=replay_capacity,
            replay_alpha=replay_alpha,
            replay_temp=replay_temp,
            replay_action_ce_coef=replay_action_ce_coef,
            replay_k=replay_k,
            main_step=main_step,
            rr_ptr=rr_ptr,
            step_cap_hit=step_cap_hit,
            k_curr=k_curr,
            loss_ema=loss_ema,
            k_steps=k_steps,
            probe_left=probe_left,
            peak_ema=peak_ema,
            target_ema=target_ema,
            phase=curr_phase,
            replay_iter_done=replay_iter_done,
            replay_buffer=replay_buffer,
            replay_best_score=replay_best_score,
        )
        latest_path, step_path, removed = _save_checkpoint(
            checkpoint_path=checkpoint_path,
            payload=payload,
            phase=curr_phase,
            step=curr_step,
            keep_last=keep_last_checkpoints,
        )
        msg = f'[checkpoint] saved latest={latest_path} step_copy={step_path}'
        if removed:
            msg += f' removed_old={len(removed)}'
        print(msg)

    if phase == 'main':
        while True:
            active_indices = [st.index for st in board_states if not st.graduated]
            if not active_indices:
                print(f'[main] all boards graduated at step={main_step}; enter replay/RL phase.')
                break

            if int(steps) > 0 and main_step >= int(steps):
                step_cap_hit = True
                print(
                    f'[main] reached steps={int(steps)} with {len(active_indices)} active boards left; '
                    'stop main training and continue to replay/RL.'
                )
                break

            if int(max_main_steps) > 0 and main_step >= int(max_main_steps):
                step_cap_hit = True
                print(
                    f'[main] reached max_main_steps={int(max_main_steps)} with {len(active_indices)} active boards left; '
                    'stop main training and continue to replay/RL.'
                )
                break

            idx = active_indices[rr_ptr % len(active_indices)]
            rr_ptr += 1
            main_step += 1

            if adaptive_k:
                k = int(k_curr)
            else:
                k = schedule_kmax(main_step, schedule_total_steps, warmup, kmax_final)

            alpha_total = int(expert_mix_anneal_steps) if int(expert_mix_anneal_steps) > 0 else schedule_total_steps
            alpha_hold = max(int(warmup), int(0.25 * max(1, alpha_total)))
            alpha = schedule_mix(main_step, alpha_total, alpha_hold, expert_mix_start, expert_mix_end)

            task_item = train_tasks[idx]
            board_state = board_states[idx]
            task = task_from_json(task_item['path'])
            env = PlacementEnv(task, **env_kwargs)
            expert_actions = task_item['expert_actions']
            k_eff = min(int(k), int(board_state.kmax))

            opt.zero_grad(set_to_none=True)
            loss = rollout_suffix_loss(
                model, env, expert_actions, task_item['path'], k_eff, alpha, teacher, region_cfg, geom_cfg, device_t,
                do_backward=True,
                grad_weight=1.0,
            )
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_val = float(loss.item())
            is_full_board = int(k_eff) >= int(board_state.kmax)

            if is_full_board:
                graduated, grad_reason = _update_board_full_state(
                    board_state,
                    loss_val=loss_val,
                    main_step=main_step,
                    ema_beta=float(board_full_ema_beta),
                    probe_visits=int(board_full_probe_visits),
                    drop_ratio=float(board_full_drop_ratio),
                    min_visits=int(board_full_min_visits),
                    plateau_patience=int(board_full_plateau_patience),
                    plateau_rel_change=float(board_full_plateau_rel_change),
                    max_visits=int(board_full_max_visits),
                )
                if graduated:
                    fp = float(board_state.full_peak_ema) if board_state.full_peak_ema is not None else float('nan')
                    ft = float(board_state.full_target_ema) if board_state.full_target_ema is not None else float('nan')
                    fe = float(board_state.full_loss_ema) if board_state.full_loss_ema is not None else float('nan')
                    print(
                        f'[graduate] board={board_state.path} kmax={board_state.kmax} '
                        f'full_visits={board_state.full_visits} full_ema={fe:.4f} peak={fp:.4f} target={ft:.4f} '
                        f'reason={grad_reason} step={main_step}'
                    )

            growth_states = [
                st for st in board_states
                if (not st.graduated) and (int(st.kmax) > int(k_curr))
            ]

            if adaptive_k and (not is_full_board) and (int(board_state.kmax) > int(k_curr)):
                drop = float(adaptive_k_drop_ratio_first) if int(k_curr) <= 1 else float(adaptive_k_drop_ratio)
                ready, ready_reason = _update_board_k_stage_state(
                    board_state,
                    k_stage=int(k_curr),
                    loss_val=loss_val,
                    ema_beta=float(adaptive_k_ema_beta),
                    probe_steps=int(adaptive_k_probe_steps),
                    drop_ratio=drop,
                    min_visits=int(adaptive_k_min_steps),
                    max_visits=int(adaptive_k_max_steps_per_k),
                    max_visits_target_slack=float(adaptive_k_max_visits_target_slack),
                    outlier_release_factor=float(adaptive_k_outlier_release_factor),
                )
                if ready:
                    ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                    kp = float(board_state.k_stage_peak_ema) if board_state.k_stage_peak_ema is not None else float('nan')
                    kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                    print(
                        f'[k-ready] board={board_state.path} k={int(board_state.k_stage)} '
                        f'visits={board_state.k_stage_visits} ema={ke:.4f} peak={kp:.4f} target={kt:.4f} '
                        f'reason={ready_reason} step={main_step}'
                    )
                elif ready_reason == 'max_visits_hold':
                    ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                    kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                    print(
                        f'[k-hold] board={board_state.path} k={int(board_state.k_stage)} '
                        f'visits={board_state.k_stage_visits} ema={ke:.4f} target={kt:.4f} '
                        f'reason=max_visits_hold step={main_step}'
                    )
                elif ready_reason == 'outlier_deferred':
                    ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                    kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                    print(
                        f'[k-defer] board={board_state.path} k={int(board_state.k_stage)} '
                        f'visits={board_state.k_stage_visits} ema={ke:.4f} target={kt:.4f} '
                        f'reason=outlier_release step={main_step}'
                    )

            can_promote = False
            effective_ready_cnt = 0
            deferred_cnt = 0
            ready_ratio = 0.0
            if adaptive_k and growth_states and int(k_curr) < int(kmax_final):
                warmup_done = int(main_step) >= max(0, int(warmup))
                ready_cnt = sum(
                    1 for st in growth_states
                    if int(st.k_stage) == int(k_curr) and bool(st.k_stage_ready)
                )
                deferred_cnt = sum(
                    1 for st in growth_states
                    if int(st.k_stage) == int(k_curr) and bool(st.k_stage_deferred)
                )
                effective_ready_cnt = int(ready_cnt + deferred_cnt)
                ready_ratio = float(effective_ready_cnt) / float(max(1, len(growth_states)))
                can_promote = warmup_done and ready_ratio >= float(_clip01(adaptive_k_ready_ratio))

            if can_promote:
                old_k = int(k_curr)
                k_curr = min(int(kmax_final), old_k + 1)
                k_steps = 0
                probe_left = 0
                peak_ema = None
                target_ema = None
                for st in board_states:
                    if (not st.graduated) and (int(st.kmax) > old_k):
                        _reset_board_k_stage(st, int(k_curr))
                print(
                    f'[k-adapt] promote k {old_k}->{int(k_curr)} at step={main_step} '
                    f'(effective_ready={effective_ready_cnt}/{len(growth_states)} deferred={deferred_cnt} '
                    f'ready_ratio={ready_ratio:.3f} threshold={float(_clip01(adaptive_k_ready_ratio)):.3f} warmup_done={warmup_done})'
                )

            if main_step % 200 == 0:
                active_left = sum(0 if st.graduated else 1 for st in board_states)
                grad_cnt = len(board_states) - active_left
                if adaptive_k:
                    growth_states = [
                        st for st in board_states
                        if (not st.graduated) and (int(st.kmax) > int(k_curr))
                    ]
                    ready_cnt = sum(1 for st in growth_states if int(st.k_stage) == int(k_curr) and bool(st.k_stage_ready))
                    deferred_cnt = sum(1 for st in growth_states if int(st.k_stage) == int(k_curr) and bool(st.k_stage_deferred))
                    effective_ready_cnt = int(ready_cnt + deferred_cnt)
                    ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                    kp = float(board_state.k_stage_peak_ema) if board_state.k_stage_peak_ema is not None else float('nan')
                    kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                    print(
                        f'step={main_step} active={active_left}/{len(board_states)} graduated={grad_cnt} '
                        f'board={board_state.path} k={k} k_eff={k_eff} alpha_expert={alpha:.3f} '
                        f'loss={loss_val:.4f} ready={effective_ready_cnt}/{len(growth_states)} (strict={ready_cnt} deferred={deferred_cnt}) '
                        f'board_k_visits={board_state.k_stage_visits} board_k_ema={ke:.4f} peak={kp:.4f} target={kt:.4f}'
                    )
                else:
                    print(
                        f'step={main_step} active={active_left}/{len(board_states)} graduated={grad_cnt} '
                        f'board={board_state.path} k={k} k_eff={k_eff} alpha_expert={alpha:.3f} loss={loss_val:.4f}'
                    )

                if is_full_board:
                    fp = float(board_state.full_peak_ema) if board_state.full_peak_ema is not None else float('nan')
                    ft = float(board_state.full_target_ema) if board_state.full_target_ema is not None else float('nan')
                    fe = float(board_state.full_loss_ema) if board_state.full_loss_ema is not None else float('nan')
                    print(
                        f'          full-board visits={board_state.full_visits} full_ema={fe:.4f} '
                        f'peak={fp:.4f} target={ft:.4f}'
                    )

            if checkpoint_path and int(checkpoint_every_steps) > 0 and (main_step % int(checkpoint_every_steps) == 0):
                _checkpoint_now('main', main_step)

            if stop_requested['signal'] is not None:
                print(f"[signal] graceful stop after main step={main_step}")
                _checkpoint_now('main', main_step)
                return

    # Weighted replay fine-tune
    phase = 'replay'
    if replay_finetune:
        print('=== weighted replay fine-tune ===')
        rb = replay_buffer if replay_buffer is not None else WeightedReplayBuffer(capacity=replay_capacity, alpha=replay_alpha)
        replay_k_eff = int(replay_k) if replay_k is not None else int(Tmax)

        if replay_buffer is None or len(rb) == 0:
            # bootstrap with expert trajectories (optional; gives stable starting points)
            for it in train_tasks:
                task = task_from_json(it['path'])
                env = PlacementEnv(task, **env_kwargs)
                w2, h2 = env.grid_shape()
                R2 = len(env.rotations)
                flat = [_flatten_action(a[0], a[1], a[2], w2, h2, R2) for a in it['expert_actions']]
                env2 = PlacementEnv(task, **env_kwargs)
                for a in it['expert_actions']:
                    _o, _r, _d, info = env2.step(a)
                    if info.get('illegal'):
                        break
                score = -float(env2.prev_obj)
                rb.add({'task_path': it['path'], 'actions_flat': flat, 'score': score}, priority=math.exp(score / max(1e-6, replay_temp)))

        best_score = float(replay_best_score) if replay_best_score is not None else (max([it.get('score', -1e9) for it in rb.items]) if len(rb) else -1e9)

        for it_i in range(replay_iter_done, replay_iters):
            for _ in range(replay_rollouts_per_iter):
                task_item = random.choice(train_tasks)
                ep = rollout_episode(model, task_item['path'], device_t, teacher, region_cfg, env_kwargs=env_kwargs)
                if (not ep['terminated']) and ep['steps'] > 0:
                    p = math.exp(float(ep['score']) / max(1e-6, replay_temp))
                    rb.add(ep, priority=p)
                    best_score = max(best_score, float(ep['score']))

            upd_losses: List[float] = []
            for _ in range(replay_update_steps):
                batch = rb.sample(replay_batch_size)
                if not batch:
                    continue

                scores = np.array([float(ep.get('score', 0.0)) for ep in batch], dtype=np.float64)
                wts = np.exp(scores / max(1e-6, float(replay_temp)))
                wts = wts / max(1e-12, float(wts.mean()))

                opt.zero_grad(set_to_none=True)

                loss_b_val = 0.0
                for ep, wi in zip(batch, wts.tolist()):
                    loss_ep = replay_teacher_loss_on_episode(
                        model,
                        ep,
                        k=replay_k_eff,
                        teacher=teacher,
                        region_cfg=region_cfg,
                        geom_cfg=geom_cfg,
                        expert_actions_by_path=expert_actions_by_path,
                        device=device_t,
                        action_ce_coef=float(replay_action_ce_coef),
                        do_backward=True,
                        grad_weight=float(wi) / max(1, len(batch)),
                    )
                    loss_b_val += float(wi) * float(loss_ep.item())

                loss_b_val = loss_b_val / max(1, len(batch))
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                upd_losses.append(float(loss_b_val))

            replay_iter_done = int(it_i + 1)
            replay_buffer = rb
            replay_best_score = float(best_score)

            if upd_losses:
                print(
                    f'replay iter={it_i + 1}/{replay_iters} buffer={len(rb)} '
                    f'best_score={best_score:.4f} upd_loss={float(np.mean(upd_losses)):.4f}'
                )

            if checkpoint_path:
                _checkpoint_now('replay', main_step)

            if stop_requested['signal'] is not None:
                print(f"[signal] graceful stop during replay iter={it_i + 1}")
                _checkpoint_now('replay', main_step)
                return

    final_payload = _build_checkpoint_payload(
        model=model,
        opt=opt,
        obs_dim=obs_dim,
        max_tokens=max_tokens,
        board_states=board_states,
        kmax_final=kmax_final,
        warmup=warmup,
        steps=steps,
        max_main_steps=max_main_steps,
        adaptive_k=adaptive_k,
        adaptive_k_drop_ratio=adaptive_k_drop_ratio,
        adaptive_k_drop_ratio_first=adaptive_k_drop_ratio_first,
        adaptive_k_probe_steps=adaptive_k_probe_steps,
        adaptive_k_min_steps=adaptive_k_min_steps,
        adaptive_k_ema_beta=adaptive_k_ema_beta,
        adaptive_k_max_steps_per_k=adaptive_k_max_steps_per_k,
        board_full_ema_beta=board_full_ema_beta,
        board_full_probe_visits=board_full_probe_visits,
        board_full_drop_ratio=board_full_drop_ratio,
        board_full_min_visits=board_full_min_visits,
        board_full_plateau_patience=board_full_plateau_patience,
        board_full_plateau_rel_change=board_full_plateau_rel_change,
        board_full_max_visits=board_full_max_visits,
        env_alignment_bonus=env_alignment_bonus,
        env_edge_bonus=env_edge_bonus,
        env_edge_eps_mm=env_edge_eps_mm,
        reward_non_interface_edge_penalty=reward_non_interface_edge_penalty,
        reward_non_interface_edge_margin_mm=reward_non_interface_edge_margin_mm,
        reward_density_penalty=reward_density_penalty,
        reward_density_radius_mm=reward_density_radius_mm,
        reward_interior_penalty=reward_interior_penalty,
        reward_interior_margin_ratio=reward_interior_margin_ratio,
        objective_nslw_weight=objective_nslw_weight,
        objective_region_weight=objective_region_weight,
        objective_conn_weight=objective_conn_weight,
        objective_align_weight=objective_align_weight,
        objective_group_weight=objective_group_weight,
        objective_anchor_weight=objective_anchor_weight,
        objective_boundary_group_weight=objective_boundary_group_weight,
        objective_pitch_weight=objective_pitch_weight,
        objective_orientation_weight=objective_orientation_weight,
        objective_edge_clearance_weight=objective_edge_clearance_weight,
        objective_interior_weight=objective_interior_weight,
        objective_density_weight=objective_density_weight,
        objective_soft_spacing_weight=objective_soft_spacing_weight,
        objective_neatness_weight=objective_neatness_weight,
        edge_band_ratio=edge_band_ratio,
        edge_band_center_ratio=edge_band_center_ratio,
        soft_spacing_same_group_extra_mm=soft_spacing_same_group_extra_mm,
        soft_spacing_cross_group_extra_mm=soft_spacing_cross_group_extra_mm,
        soft_spacing_large_extra_mm=soft_spacing_large_extra_mm,
        same_group_density_scale=same_group_density_scale,
        critical_neighbor_density_scale=critical_neighbor_density_scale,
        anchor_group_density_scale=anchor_group_density_scale,
        large_pair_density_scale=large_pair_density_scale,
        teacher=teacher,
        region_cfg=region_cfg,
        geom_cfg=geom_cfg,
        expert_mix_start=expert_mix_start,
        expert_mix_end=expert_mix_end,
        expert_mix_anneal_steps=expert_mix_anneal_steps,
        replay_finetune=replay_finetune,
        replay_iters=replay_iters,
        replay_rollouts_per_iter=replay_rollouts_per_iter,
        replay_update_steps=replay_update_steps,
        replay_batch_size=replay_batch_size,
        replay_capacity=replay_capacity,
        replay_alpha=replay_alpha,
        replay_temp=replay_temp,
        replay_action_ce_coef=replay_action_ce_coef,
        replay_k=replay_k,
        main_step=main_step,
        rr_ptr=rr_ptr,
        step_cap_hit=step_cap_hit,
        k_curr=k_curr,
        loss_ema=loss_ema,
        k_steps=k_steps,
        probe_left=probe_left,
        peak_ema=peak_ema,
        target_ema=target_ema,
        phase='done',
        replay_iter_done=replay_iter_done,
        replay_buffer=replay_buffer,
        replay_best_score=replay_best_score,
    )
    _atomic_torch_save(final_payload, save_path)
    if checkpoint_path:
        _checkpoint_now('done', main_step)
    print(f'saved: {save_path}')

