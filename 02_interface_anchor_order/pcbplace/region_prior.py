from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .utils import load_json

REGION_TYPE_NAMES: Tuple[str, ...] = (
    "edge_top",
    "edge_bottom",
    "edge_left",
    "edge_right",
    "core",
    "free",
)
SEMANTIC_CLASS_NAMES: Tuple[str, ...] = (
    "core",
    "power",
    "interface",
    "power_support",
    "interface_support",
    "passive",
    "support",
    "clock",
    "ui",
    "rf",
    "mechanical",
    "other",
)
SIDE_PREFERENCE_NAMES: Tuple[str, ...] = ("free", "left", "right", "top", "bottom")
SUBZONE_NAMES: Tuple[str, ...] = ("free", "around", "left", "right", "top", "bottom")
PAIRWISE_RELATION_NAMES: Tuple[str, ...] = ("same_group", "anchor_neighbor", "critical_neighbor")

_REGION_TYPE_TO_IDX = {name: i for i, name in enumerate(REGION_TYPE_NAMES)}
_SEMANTIC_CLASS_TO_IDX = {name: i for i, name in enumerate(SEMANTIC_CLASS_NAMES)}
_SIDE_PREFERENCE_TO_IDX = {name: i for i, name in enumerate(SIDE_PREFERENCE_NAMES)}
_SUBZONE_TO_IDX = {name: i for i, name in enumerate(SUBZONE_NAMES)}


@dataclass
class RegionPriorConfig:
    enabled: bool = True
    grid_x: int = 6
    grid_y: int = 6
    heatmap_sigma_cells: float = 0.85
    zone_edge_ratio: float = 0.12
    zone_core_ratio: float = 0.28
    zone_prior_weight: float = 0.35
    aux_heatmap_weight: float = 0.30
    aux_zone_weight: float = 0.10
    aux_side_weight: float = 0.10
    aux_subzone_weight: float = 0.08
    aux_pairwise_weight: float = 0.08




def semantic_review_weight(
    review_status: str,
    auto_confidence: float,
    needs_review: bool = False,
) -> float:
    """Map review state + auto confidence to a training trust weight in [0,1].

    Intuition:
      - approved / edited labels are treated as high-trust human-checked supervision
      - seeded / unreviewed labels are softly trusted according to auto_confidence
      - rejected / ignored labels contribute no semantic supervision
      - explicit needs_review slightly downweights seeded labels further
    """
    status = str(review_status or "seeded").strip().lower()
    try:
        conf = float(auto_confidence)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    if status in {"approved", "edited", "accepted", "confirmed", "manual_reviewed"}:
        weight = 1.0
    elif status in {"rejected", "ignore", "ignored", "discarded"}:
        weight = 0.0
    elif status in {"seeded", "auto_seeded", "untracked", "", "pending"}:
        weight = 0.35 + 0.65 * conf
    else:
        weight = 0.50 + 0.50 * conf

    if bool(needs_review) and weight < 0.999:
        weight *= 0.85
    return float(max(0.0, min(1.0, weight)))

def semantic_class_to_index(name: str) -> int:
    key = str(name or "other").strip().lower()
    return int(_SEMANTIC_CLASS_TO_IDX.get(key, _SEMANTIC_CLASS_TO_IDX["other"]))


def region_type_to_index(name: str) -> int:
    key = str(name or "free").strip().lower()
    return int(_REGION_TYPE_TO_IDX.get(key, _REGION_TYPE_TO_IDX["free"]))


def side_preference_to_index(name: str) -> int:
    key = str(name or "free").strip().lower()
    if key.startswith("edge_"):
        key = key.split("_", 1)[1]
    return int(_SIDE_PREFERENCE_TO_IDX.get(key, _SIDE_PREFERENCE_TO_IDX["free"]))


def subzone_to_index(name: str) -> int:
    key = str(name or "free").strip().lower()
    return int(_SUBZONE_TO_IDX.get(key, _SUBZONE_TO_IDX["free"]))


def _swap_size_if_needed(size_mm: Tuple[float, float], rot_deg: int) -> Tuple[float, float]:
    w, h = float(size_mm[0]), float(size_mm[1])
    if int(rot_deg) % 180 != 0:
        w, h = h, w
    return w, h


def _bbox_from_pose(x: float, y: float, size_mm: Tuple[float, float], rot_deg: int) -> Tuple[float, float, float, float]:
    w, h = _swap_size_if_needed(size_mm, rot_deg)
    return (x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h)


def classify_region_type_from_bbox(
    task_bbox_mm: Tuple[float, float, float, float],
    bb: Tuple[float, float, float, float],
    zone_edge_ratio: float,
    zone_core_ratio: float,
) -> int:
    xmin, ymin, xmax, ymax = task_bbox_mm
    a, b, c, d = bb
    clearance = min(a - xmin, xmax - c, b - ymin, ymax - d)
    span = max(1e-6, min(xmax - xmin, ymax - ymin))
    ratio = float(clearance) / float(span)
    if ratio <= float(zone_edge_ratio):
        dists = {
            "edge_left": a - xmin,
            "edge_right": xmax - c,
            "edge_bottom": b - ymin,
            "edge_top": ymax - d,
        }
        return region_type_to_index(min(dists, key=dists.get))
    if ratio >= float(zone_core_ratio):
        return region_type_to_index("core")
    return region_type_to_index("free")


@lru_cache(maxsize=4096)
def load_region_targets_for_task(
    task_path: str,
    grid_x: int,
    grid_y: int,
    sigma_cells: float,
    zone_edge_ratio: float,
    zone_core_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    data = load_json(task_path)
    board = data["board"]
    bbox = tuple(board["bbox_mm"])
    out: Dict[str, Dict[str, Any]] = {}
    for comp in data.get("components", []):
        expert = comp.get("expert") or {}
        if "xy_mm" not in expert:
            continue
        ref = str(comp["ref"])
        x, y = float(expert["xy_mm"][0]), float(expert["xy_mm"][1])
        rot = int(expert.get("rot", 0))
        size_mm = tuple(comp.get("size_mm", [1.0, 1.0]))
        bb = _bbox_from_pose(x, y, size_mm=size_mm, rot_deg=rot)
        region_type = comp.get("region_type") or (comp.get("semantic") or {}).get("region_type") or "free"
        semantic_class = comp.get("semantic_class") or (comp.get("semantic") or {}).get("semantic_class") or "other"
        side_preference = comp.get("side_preference") or (comp.get("semantic") or {}).get("side_preference") or "free"
        subzone = comp.get("subzone") or (comp.get("semantic") or {}).get("subzone") or "free"
        same_side_group = comp.get("same_side_group") or (comp.get("semantic") or {}).get("same_side_group") or None
        anchor_ref = comp.get("anchor_ref") or (comp.get("semantic") or {}).get("anchor_ref") or None
        critical_neighbors = tuple(comp.get("critical_neighbors") or (comp.get("semantic") or {}).get("critical_neighbors") or [])
        functional_group = comp.get("functional_group") or (comp.get("semantic") or {}).get("functional_group") or "misc"
        review = comp.get("semantic_review") or {}
        review_status = str(review.get("review_status") or "seeded")
        auto_confidence = float(review.get("auto_confidence", 1.0) if review.get("auto_confidence", None) not in (None, "") else 1.0)
        needs_review = bool(review.get("needs_review", False))
        review_weight = semantic_review_weight(review_status, auto_confidence, needs_review)
        # Fallback keeps compatibility with old data lacking step-1 semantic labels.
        if str(region_type).strip().lower() not in _REGION_TYPE_TO_IDX:
            region_type = REGION_TYPE_NAMES[classify_region_type_from_bbox(bbox, bb, zone_edge_ratio, zone_core_ratio)]
        out[ref] = {
            "region_type": region_type_to_index(region_type),
            "semantic_class": semantic_class_to_index(semantic_class),
            "side_preference": side_preference_to_index(side_preference),
            "subzone": subzone_to_index(subzone),
            "same_side_group": str(same_side_group or ""),
            "anchor_ref": str(anchor_ref or ""),
            "critical_neighbors": tuple(str(v) for v in critical_neighbors),
            "functional_group": str(functional_group or "misc"),
            "xy_mm": (float(x), float(y)),
            "rot": int(rot),
            "review_status": review_status,
            "auto_confidence": auto_confidence,
            "needs_review": needs_review,
            "review_weight": review_weight,
        }
    return out


def build_action_region_indices(
    env: Any,
    ref: str,
    grid_x: int,
    grid_y: int,
    zone_edge_ratio: float,
    zone_core_ratio: float,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    w_cells, h_cells = env.grid_shape()
    grid_mm = float(env.task.grid_mm)
    R = len(env.rotations)
    comp = env.comp_by_ref[ref]

    ix = np.arange(w_cells, dtype=np.float32)[:, None]
    iy = np.arange(h_cells, dtype=np.float32)[None, :]
    X = xmin + (ix + 0.5) * grid_mm
    Y = ymin + (iy + 0.5) * grid_mm
    X = np.broadcast_to(X, (w_cells, h_cells)).astype(np.float32)
    Y = np.broadcast_to(Y, (w_cells, h_cells)).astype(np.float32)

    region_idx = np.zeros((R, w_cells, h_cells), dtype=np.int64)
    for ri, rot in enumerate(env.rotations):
        w_mm, h_mm = _swap_size_if_needed(comp.size_mm, int(rot))
        a = X - 0.5 * w_mm
        b = Y - 0.5 * h_mm
        c = X + 0.5 * w_mm
        d = Y + 0.5 * h_mm
        clearance = np.minimum(np.minimum(a - xmin, xmax - c), np.minimum(b - ymin, ymax - d))
        span = max(1e-6, min(xmax - xmin, ymax - ymin))
        ratio = clearance / span
        dstack = np.stack([ymax - d, b - ymin, a - xmin, xmax - c], axis=0)
        nearest = np.argmin(dstack, axis=0)
        names = np.array([
            region_type_to_index("edge_top"),
            region_type_to_index("edge_bottom"),
            region_type_to_index("edge_left"),
            region_type_to_index("edge_right"),
        ], dtype=np.int64)
        region = np.full((w_cells, h_cells), region_type_to_index("free"), dtype=np.int64)
        edge_mask = ratio <= float(zone_edge_ratio)
        core_mask = ratio >= float(zone_core_ratio)
        region[edge_mask] = names[nearest[edge_mask]]
        region[core_mask] = region_type_to_index("core")
        region_idx[ri] = region
    return region_idx.reshape(-1)


def action_region_prior_from_predictions(
    region_type_logits: torch.Tensor,
    region_type_idx_flat: np.ndarray,
    zone_prior_weight: float,
) -> np.ndarray:
    region_logp = torch.log_softmax(region_type_logits.detach(), dim=-1).cpu().numpy().reshape(-1)
    prior = float(zone_prior_weight) * region_logp[region_type_idx_flat]
    prior = prior - float(np.max(prior))
    return prior.astype(np.float32)
