from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .env import PlacementEnv
from .env_cuda import (
    action_region_prior_from_predictions_cuda,
    build_action_region_indices_cuda,
    build_teacher_distribution_cuda,
)
from .model import MaskedPolicy
from .region_prior import RegionPriorConfig


ACTION_SCORING_VERSION = "objective_aware_v1"
DEFAULT_ROLLOUT_OBJECTIVE_ALPHA = 0.35
DEFAULT_ROLLOUT_REGION_ALPHA = 0.10
DEFAULT_ROLLOUT_ACTION_PRIOR_ALPHA = 1.0


def masked_zscore(
    values: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    """Z-score values over legal candidates only."""
    out = torch.zeros_like(values)
    legal_bool = legal.to(dtype=torch.bool)
    legal_values = values[legal_bool]
    if legal_values.numel() == 0:
        return out
    mean = legal_values.mean()
    std = legal_values.std(unbiased=False).clamp_min(1e-6)
    out[legal_bool] = (legal_values - mean) / std
    return out


def masked_zscore_batched(
    values: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    """Vectorized row-wise z-score over legal candidates.

    This is numerically equivalent to calling :func:`masked_zscore` for every
    row, but keeps the rollout/replay hot path on the device instead of
    bouncing through a Python loop for each batch row.  Rows with no legal
    candidates return zeros, matching the single-row helper.
    """
    legal_bool = legal.to(dtype=torch.bool, device=values.device)
    values_t = values.to(device=values.device)
    if values_t.dim() == 1:
        return masked_zscore(values_t, legal_bool)
    if legal_bool.shape != values_t.shape:
        legal_bool = legal_bool.reshape_as(values_t)

    dtype = values_t.dtype
    counts = legal_bool.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
    masked_values = values_t.masked_fill(~legal_bool, 0.0)
    mean = masked_values.sum(dim=1, keepdim=True) / counts
    centered = torch.where(legal_bool, values_t - mean, torch.zeros_like(values_t))
    var = (centered * centered).sum(dim=1, keepdim=True) / counts
    std = var.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
    return torch.where(legal_bool, centered / std, torch.zeros_like(values_t))


def _teacher_candidate_mask_batched(
    mask_batch: torch.Tensor,
    objective_residual_batch: torch.Tensor,
    teacher: Any,
    *,
    region_prior_batch: Optional[torch.Tensor] = None,
    action_prior_batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized candidate gating equivalent to build_teacher_distribution_cuda."""
    legal = mask_batch > 0.5
    candidate = legal.clone()
    if not bool(getattr(teacher, "gate_rollout", False)):
        return candidate

    energy = objective_residual_batch.clone()
    if region_prior_batch is not None:
        energy = energy - float(getattr(teacher, "lambda_region_prior", 0.0)) * (
            region_prior_batch.to(
                device=energy.device,
                dtype=energy.dtype,
            )
        )
    if action_prior_batch is not None:
        energy = energy - action_prior_batch.to(
            device=energy.device,
            dtype=energy.dtype,
        )

    objective_delta_max = getattr(teacher, "objective_delta_max", None)
    if objective_delta_max is not None:
        thresholded = candidate & (
            objective_residual_batch <= float(objective_delta_max)
        )
        has_any = thresholded.any(dim=1, keepdim=True)
        candidate = torch.where(has_any, thresholded, candidate)

    topk = int(getattr(teacher, "topk", 0) or 0)
    if topk > 0 and candidate.shape[1] > topk:
        counts = candidate.sum(dim=1)
        energy_for_topk = energy.masked_fill(~candidate, float("inf"))
        _, top_indices = torch.topk(
            energy_for_topk,
            k=topk,
            dim=1,
            largest=False,
        )
        candidate_top = torch.zeros_like(candidate)
        candidate_top.scatter_(1, top_indices, True)
        candidate_top = candidate_top & candidate
        candidate = torch.where(
            (counts > topk).unsqueeze(1),
            candidate_top,
            candidate,
        )
    return candidate


@torch.no_grad()
def score_actions(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    objective_residual: torch.Tensor,
    objective_total: torch.Tensor,
    region_prior: Optional[torch.Tensor],
    action_prior: Optional[torch.Tensor],
    *,
    teacher: Any,
    objective_alpha: float = DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    region_alpha: float = DEFAULT_ROLLOUT_REGION_ALPHA,
    action_prior_alpha: float = DEFAULT_ROLLOUT_ACTION_PRIOR_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical objective-aware action scoring for one board.

    This is the single source of truth used by main-training rollout, replay
    collection, greedy inference, and beam inference.
    """
    device = logits.device
    legal = mask.reshape(-1).to(device=device) > 0.5
    base_score = (
        logits.reshape(-1)
        + bias.reshape(-1).to(device=device, dtype=logits.dtype)
    ).masked_fill(~legal, -1e9)

    region_prior_t: Optional[torch.Tensor] = None
    if region_prior is not None:
        region_prior_t = torch.as_tensor(
            region_prior,
            device=device,
            dtype=logits.dtype,
        ).reshape(-1)

    action_prior_t: Optional[torch.Tensor] = None
    if action_prior is not None:
        action_prior_t = torch.as_tensor(
            action_prior,
            device=device,
            dtype=logits.dtype,
        ).reshape(-1)

    candidate = legal
    if bool(getattr(teacher, "gate_rollout", False)):
        _q, candidate = build_teacher_distribution_cuda(
            mask.reshape(-1).to(device=device),
            objective_residual.reshape(-1).to(
                device=device,
                dtype=logits.dtype,
            ),
            teacher,
            device=device,
            region_flat=region_prior_t,
            action_prior_flat=action_prior_t,
        )
        base_score = base_score.masked_fill(~candidate, -1e9)

    legal_for_norm = candidate & torch.isfinite(base_score)
    objective_z = masked_zscore(
        objective_total.reshape(-1).to(
            device=device,
            dtype=logits.dtype,
        ),
        legal_for_norm,
    )
    score = base_score - float(objective_alpha) * objective_z

    if action_prior_t is not None and float(action_prior_alpha) != 0.0:
        action_prior_z = masked_zscore(
            action_prior_t,
            legal_for_norm,
        )
        score = score + float(action_prior_alpha) * action_prior_z

    if region_prior_t is not None and float(region_alpha) != 0.0:
        region_prior_z = masked_zscore(
            region_prior_t,
            legal_for_norm,
        )
        score = score + float(region_alpha) * region_prior_z

    score = score.masked_fill(~legal_for_norm, -1e9)
    return score, legal_for_norm


def compose_action_scores_batched(
    logits_batch: torch.Tensor,
    bias_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    objective_residual_batch: torch.Tensor,
    objective_total_batch: torch.Tensor,
    region_prior_batch: Optional[torch.Tensor],
    action_prior_batch: Optional[torch.Tensor],
    *,
    teacher: Any,
    objective_alpha: float = DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    region_alpha: float = DEFAULT_ROLLOUT_REGION_ALPHA,
    action_prior_alpha: float = DEFAULT_ROLLOUT_ACTION_PRIOR_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose canonical batched scores while preserving logits gradients."""
    legal = mask_batch > 0.5
    score = (
        logits_batch
        + bias_batch.to(
            device=logits_batch.device,
            dtype=logits_batch.dtype,
        )
    ).masked_fill(~legal, -1e9)

    with torch.no_grad():
        candidate = _teacher_candidate_mask_batched(
            mask_batch,
            objective_residual_batch,
            teacher,
            region_prior_batch=region_prior_batch,
            action_prior_batch=action_prior_batch,
        )
    score = score.masked_fill(~candidate, -1e9)
    legal_for_norm = candidate & torch.isfinite(score)

    objective_z = masked_zscore_batched(
        objective_total_batch.to(device=score.device, dtype=score.dtype),
        legal_for_norm,
    )
    score = score - float(objective_alpha) * objective_z

    if action_prior_batch is not None and float(action_prior_alpha) != 0.0:
        action_prior_z = masked_zscore_batched(
            action_prior_batch.to(device=score.device, dtype=score.dtype),
            legal_for_norm,
        )
        score = score + float(action_prior_alpha) * action_prior_z

    if region_prior_batch is not None and float(region_alpha) != 0.0:
        region_prior_z = masked_zscore_batched(
            region_prior_batch.to(device=score.device, dtype=score.dtype),
            legal_for_norm,
        )
        score = score + float(region_alpha) * region_prior_z

    return score.masked_fill(~legal_for_norm, -1e9), legal_for_norm


@torch.no_grad()
def score_actions_batched(
    logits_batch: torch.Tensor,
    bias_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    objective_residual_batch: torch.Tensor,
    objective_total_batch: torch.Tensor,
    region_prior_batch: Optional[torch.Tensor],
    action_prior_batch: Optional[torch.Tensor],
    *,
    teacher: Any,
    objective_alpha: float = DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    region_alpha: float = DEFAULT_ROLLOUT_REGION_ALPHA,
    action_prior_alpha: float = DEFAULT_ROLLOUT_ACTION_PRIOR_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized canonical action scoring for padded multi-board batches."""
    return compose_action_scores_batched(
        logits_batch,
        bias_batch,
        mask_batch,
        objective_residual_batch,
        objective_total_batch,
        region_prior_batch,
        action_prior_batch,
        teacher=teacher,
        objective_alpha=objective_alpha,
        region_alpha=region_alpha,
        action_prior_alpha=action_prior_alpha,
    )


def policy_log_probs_batched(
    logits_batch: torch.Tensor,
    bias_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    objective_residual_batch: torch.Tensor,
    objective_total_batch: torch.Tensor,
    region_prior_batch: Optional[torch.Tensor],
    action_prior_batch: Optional[torch.Tensor],
    *,
    teacher: Any,
    temperature: float = 1.0,
    objective_alpha: float = DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    region_alpha: float = DEFAULT_ROLLOUT_REGION_ALPHA,
    action_prior_alpha: float = DEFAULT_ROLLOUT_ACTION_PRIOR_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable log-probs and entropy for the shaped rollout policy.

    The distribution is computed from the same composed action scores used for
    rollout selection: model logits plus legality bias, objective shaping, region
    prior, and action prior.  This keeps replay/PG updates on-policy with the
    sampler, but it is intentionally not a pure-logit policy gradient.
    """
    temperature_value = max(1e-6, float(temperature))
    scores, candidates = compose_action_scores_batched(
        logits_batch,
        bias_batch,
        mask_batch,
        objective_residual_batch,
        objective_total_batch,
        region_prior_batch,
        action_prior_batch,
        teacher=teacher,
        objective_alpha=objective_alpha,
        region_alpha=region_alpha,
        action_prior_alpha=action_prior_alpha,
    )
    scaled_scores = scores / temperature_value
    log_probs = torch.log_softmax(scaled_scores, dim=-1)
    probs = torch.softmax(scaled_scores, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs, entropy, candidates


def flatten_action(rix: int, ix: int, iy: int, w: int, h: int) -> int:
    """Flatten ``(rotation, x, y)`` using the canonical ``[R, X, Y]`` order."""
    return int(rix) * (int(w) * int(h)) + int(ix) * int(h) + int(iy)


def unflatten_action(action: int, w: int, h: int) -> Tuple[int, int, int]:
    """Inverse of :func:`flatten_action` for the canonical ``[R, X, Y]`` order."""
    wh = int(w) * int(h)
    rix = int(action) // wh
    rem = int(action) % wh
    ix = rem // int(h)
    iy = rem % int(h)
    return int(rix), int(ix), int(iy)


def _component_nets(env: PlacementEnv, ref: str) -> set[str]:
    comp = env.comp_by_ref[ref]
    return {
        str(net)
        for net, _pad in comp.pads
        if net and str(net).upper() not in {"", "GND", "GROUND"}
    }


def build_context_ref_order(
    env: PlacementEnv,
    ref: str,
    max_tokens: int = 128,
) -> List[str]:
    """Return placed references used before the current token.

    Connected placed components are prioritized by shared-net count. The result
    is returned oldest-to-newest so callers can append the current component as
    the final token. This ordering is shared by training and inference.
    """
    current_nets = _component_nets(env, ref)
    shared = []
    other = []
    for placed_ref in reversed(list(env.placed_order)):
        shared_count = len(current_nets.intersection(_component_nets(env, placed_ref)))
        (shared if shared_count > 0 else other).append((placed_ref, shared_count))
    ordered = [
        placed_ref
        for placed_ref, _ in sorted(shared, key=lambda item: -item[1])
    ] + [placed_ref for placed_ref, _ in other]
    keep = ordered[: max(0, int(max_tokens) - 1)]
    return list(reversed(keep))


def cached_region_indices_cuda(
    env: PlacementEnv,
    ref: str,
    region_cfg: RegionPriorConfig,
    device: torch.device,
) -> torch.Tensor:
    """Build/cache action-to-region indices shared by train and inference."""
    cache = getattr(env, "_region_indices_cuda_cache", None)
    key = (
        str(device),
        str(ref),
        int(region_cfg.grid_x),
        int(region_cfg.grid_y),
        float(region_cfg.legacy_zone_edge_ratio),
        float(region_cfg.legacy_zone_core_ratio),
    )
    if isinstance(cache, dict) and key in cache:
        return cache[key]

    indices = build_action_region_indices_cuda(
        env,
        ref,
        device,
        grid_x=int(region_cfg.grid_x),
        grid_y=int(region_cfg.grid_y),
        legacy_zone_edge_ratio=float(region_cfg.legacy_zone_edge_ratio),
        legacy_zone_core_ratio=float(region_cfg.legacy_zone_core_ratio),
    )
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_region_indices_cuda_cache", cache)
    cache[key] = indices
    return indices


def policy_outputs_with_region(
    model: MaskedPolicy,
    env: PlacementEnv,
    ref: str,
    tokens_t: torch.Tensor,
    action_feat_t: torch.Tensor,
    region_cfg: RegionPriorConfig,
):
    """Run the canonical single-board policy forward used by train and infer.

    The tuple shape intentionally matches the historical training helper:
    action logits, region logits, semantic logits, side logits, subzone logits,
    pairwise logits, action region prior, encoded tokens, context ref order.
    """
    outputs = model(tokens_t, action_feat_t)
    action_logits = outputs["action_logits"]
    encoded_tokens = outputs["encoded_tokens"]
    assert action_logits is not None
    assert encoded_tokens is not None
    action_logits = action_logits.squeeze(0)

    if not bool(region_cfg.enabled):
        return (
            action_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            encoded_tokens,
            None,
        )

    region_heatmap_logits = outputs["region_heatmap_logits"]
    semantic_class_logits = outputs["semantic_class_logits"]
    side_preference_logits = outputs["side_preference_logits"]
    subzone_logits = outputs["subzone_logits"]
    pairwise_relation_logits = outputs["pairwise_relation_logits"]

    assert region_heatmap_logits is not None
    assert semantic_class_logits is not None
    assert side_preference_logits is not None
    assert subzone_logits is not None

    region_heatmap_logits = region_heatmap_logits.squeeze(0)
    semantic_class_logits = semantic_class_logits.squeeze(0)
    side_preference_logits = side_preference_logits.squeeze(0)
    subzone_logits = subzone_logits.squeeze(0)
    if pairwise_relation_logits is not None:
        pairwise_relation_logits = pairwise_relation_logits.squeeze(0)

    token_refs = build_context_ref_order(
        env,
        ref,
        max_tokens=int(tokens_t.shape[1]),
    )
    region_indices = cached_region_indices_cuda(
        env,
        ref,
        region_cfg,
        device=region_heatmap_logits.device,
    )
    region_prior = action_region_prior_from_predictions_cuda(
        region_heatmap_logits,
        action_heatmap_cell_idx_flat=region_indices,
        heatmap_action_prior_weight=float(region_cfg.heatmap_action_prior_weight),
    )
    return (
        action_logits,
        region_heatmap_logits,
        semantic_class_logits,
        side_preference_logits,
        subzone_logits,
        pairwise_relation_logits,
        region_prior,
        encoded_tokens,
        token_refs,
    )
