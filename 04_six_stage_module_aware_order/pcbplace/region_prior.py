from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
try:
    import torch
except ImportError:  # Data preparation only uses the NumPy heatmap helpers.
    torch = None

from .utils import load_json, drop_mounting_holes_from_task_json

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
    legacy_zone_edge_ratio: float = 0.12
    legacy_zone_core_ratio: float = 0.28
    heatmap_action_prior_weight: float = 0.35
    aux_heatmap_weight: float = 0.30
    aux_prior_consistency_weight: float = 0.03
    aux_semantic_weight: float = 0.10
    aux_side_weight: float = 0.10
    aux_subzone_weight: float = 0.08
    aux_pairwise_weight: float = 0.08



def region_heatmap_num_bins(grid_x: int, grid_y: int) -> int:
    return int(max(1, int(grid_x)) * max(1, int(grid_y)))


def _grid_centers(task_bbox_mm: Tuple[float, float, float, float], grid_x: int, grid_y: int) -> Tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = [float(v) for v in task_bbox_mm]
    gx = max(1, int(grid_x)); gy = max(1, int(grid_y))
    xs = np.linspace(xmin, xmax, gx + 1, dtype=np.float32)
    ys = np.linspace(ymin, ymax, gy + 1, dtype=np.float32)
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    return xc, yc


def _xy_to_grid_index(task_bbox_mm: Tuple[float, float, float, float], x: float, y: float, grid_x: int, grid_y: int) -> int:
    xmin, ymin, xmax, ymax = [float(v) for v in task_bbox_mm]
    gx = max(1, int(grid_x)); gy = max(1, int(grid_y))
    tx = 0.0 if xmax <= xmin else (float(x) - xmin) / max(1e-6, (xmax - xmin))
    ty = 0.0 if ymax <= ymin else (float(y) - ymin) / max(1e-6, (ymax - ymin))
    ix = max(0, min(gx - 1, int(math.floor(tx * gx))))
    iy = max(0, min(gy - 1, int(math.floor(ty * gy))))
    return int(ix * gy + iy)


def gaussian_heatmap_from_xy(task_bbox_mm: Tuple[float, float, float, float], x: float, y: float, grid_x: int, grid_y: int, sigma_cells: float) -> np.ndarray:
    xc, yc = _grid_centers(task_bbox_mm, grid_x, grid_y)
    X = xc[:, None]
    Y = yc[None, :]
    xmin, ymin, xmax, ymax = [float(v) for v in task_bbox_mm]
    cell_w = max(1e-6, (xmax - xmin) / max(1, int(grid_x)))
    cell_h = max(1e-6, (ymax - ymin) / max(1, int(grid_y)))
    sx = max(1e-6, float(sigma_cells) * cell_w)
    sy = max(1e-6, float(sigma_cells) * cell_h)
    heat = np.exp(-0.5 * (((X - float(x)) / sx) ** 2 + ((Y - float(y)) / sy) ** 2)).astype(np.float32)
    heat = heat.reshape(-1)
    s = float(heat.sum())
    if s <= 0.0:
        heat[:] = 1.0 / float(heat.size)
    else:
        heat /= s
    return heat.astype(np.float32)


def gaussian_heatmap_from_bbox(task_bbox_mm: Tuple[float, float, float, float], bbox_mm: Tuple[float, float, float, float], grid_x: int, grid_y: int, sigma_cells: float, confidence: float = 1.0) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox_mm]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    base = gaussian_heatmap_from_xy(task_bbox_mm, cx, cy, grid_x, grid_y, sigma_cells)
    xmin, ymin, xmax, ymax = [float(v) for v in task_bbox_mm]
    cell_w = max(1e-6, (xmax - xmin) / max(1, int(grid_x)))
    cell_h = max(1e-6, (ymax - ymin) / max(1, int(grid_y)))
    bw = max(cell_w, abs(x1 - x0))
    bh = max(cell_h, abs(y1 - y0))
    extra_sigma = 0.5 * ((bw / cell_w) + (bh / cell_h))
    spread = gaussian_heatmap_from_xy(task_bbox_mm, cx, cy, grid_x, grid_y, max(float(sigma_cells), float(extra_sigma) * 0.5))
    alpha = max(0.15, min(1.0, float(confidence)))
    heat = alpha * spread + (1.0 - alpha) * base
    heat = heat / max(1e-6, float(heat.sum()))
    return heat.astype(np.float32)


def _extract_prior_heatmap(task_bbox_mm: Tuple[float, float, float, float], comp: Dict[str, Any], grid_x: int, grid_y: int, sigma_cells: float) -> np.ndarray | None:
    prior = comp.get('prior') if isinstance(comp.get('prior'), dict) else {}
    module = comp.get('module') if isinstance(comp.get('module'), dict) else {}
    direct = prior.get('region_heatmap') or module.get('prior_region_heatmap') or comp.get('prior_region_heatmap')
    if isinstance(direct, dict) and isinstance(direct.get('values'), list):
        arr = np.asarray(direct.get('values'), dtype=np.float32).reshape(-1)
        if arr.size == region_heatmap_num_bins(grid_x, grid_y):
            sm = float(arr.sum())
            if sm > 0.0:
                return (arr / sm).astype(np.float32)
    bbox = (prior.get('region_bbox_mm') or (module.get('prior_region') or {}).get('bbox_mm') or module.get('region_bbox_mm') or comp.get('module_region_bbox'))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        conf = (prior.get('region_confidence') if prior.get('region_confidence', None) not in (None, '') else (module.get('prior_region') or {}).get('confidence', 1.0))
        try:
            conf = float(conf if conf not in (None, '') else 1.0)
        except Exception:
            conf = 1.0
        return gaussian_heatmap_from_bbox(task_bbox_mm, tuple(float(v) for v in bbox), grid_x, grid_y, sigma_cells, conf)
    center = (prior.get('region_center_mm') or (module.get('prior_region') or {}).get('center_mm'))
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return gaussian_heatmap_from_xy(task_bbox_mm, float(center[0]), float(center[1]), grid_x, grid_y, sigma_cells)
    return None


def _extract_expert_heatmap(task_bbox_mm: Tuple[float, float, float, float], comp: Dict[str, Any], grid_x: int, grid_y: int, sigma_cells: float) -> np.ndarray | None:
    expert = comp.get('expert') if isinstance(comp.get('expert'), dict) else {}
    direct = expert.get('region_heatmap') or comp.get('expert_region_heatmap')
    if isinstance(direct, dict) and isinstance(direct.get('values'), list):
        arr = np.asarray(direct.get('values'), dtype=np.float32).reshape(-1)
        if arr.size == region_heatmap_num_bins(grid_x, grid_y):
            sm = float(arr.sum())
            if sm > 0.0:
                return (arr / sm).astype(np.float32)
    xy = expert.get('xy_mm')
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return gaussian_heatmap_from_xy(task_bbox_mm, float(xy[0]), float(xy[1]), grid_x, grid_y, sigma_cells)
    bbox = expert.get('module_region_bbox_mm') or comp.get('expert_module_region_bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return gaussian_heatmap_from_bbox(task_bbox_mm, tuple(float(v) for v in bbox), grid_x, grid_y, sigma_cells, 1.0)
    return None


def _heatmap_peak_xy(task_bbox_mm: Tuple[float, float, float, float], heatmap: np.ndarray, grid_x: int, grid_y: int) -> Tuple[float, float]:
    arr = np.asarray(heatmap, dtype=np.float32).reshape(-1)
    idx = int(np.argmax(arr)) if arr.size else 0
    gx = max(1, int(grid_x)); gy = max(1, int(grid_y))
    ix = idx // gy
    iy = idx % gy
    xc, yc = _grid_centers(task_bbox_mm, gx, gy)
    return float(xc[max(0, min(gx - 1, ix))]), float(yc[max(0, min(gy - 1, iy))])



def semantic_review_weight(
    review_status: str,
    auto_confidence: float,
    needs_review: bool = False,
) -> float:
    """Map review state + auto confidence to a semantic-label trust weight in [0,1].

    Intuition:
      - approved / edited labels are treated as high-trust human-checked semantic supervision
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


def _infer_subzone_from_xy(x: float, y: float, ax: float, ay: float) -> str:
    dx = float(x) - float(ax)
    dy = float(y) - float(ay)
    eps = 1e-6
    if abs(dx) <= eps and abs(dy) <= eps:
        return "around"
    if abs(dx) >= 1.15 * abs(dy):
        return "right" if dx > 0.0 else "left"
    if abs(dy) >= 1.15 * abs(dx):
        return "top" if dy > 0.0 else "bottom"
    return "around"


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
    legacy_zone_edge_ratio: float,
    legacy_zone_core_ratio: float,
) -> int:
    xmin, ymin, xmax, ymax = task_bbox_mm
    a, b, c, d = bb
    clearance = min(a - xmin, xmax - c, b - ymin, ymax - d)
    span = max(1e-6, min(xmax - xmin, ymax - ymin))
    ratio = float(clearance) / float(span)
    if ratio <= float(legacy_zone_edge_ratio):
        dists = {
            "edge_left": a - xmin,
            "edge_right": xmax - c,
            "edge_bottom": b - ymin,
            "edge_top": ymax - d,
        }
        return region_type_to_index(min(dists, key=dists.get))
    if ratio >= float(legacy_zone_core_ratio):
        return region_type_to_index("core")
    return region_type_to_index("free")


@lru_cache(maxsize=4096)
def load_region_targets_for_task(
    task_path: str,
    grid_x: int,
    grid_y: int,
    sigma_cells: float,
    legacy_zone_edge_ratio: float,
    legacy_zone_core_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    data = drop_mounting_holes_from_task_json(load_json(task_path))
    board = data["board"]
    bbox = tuple(board["bbox_mm"])
    components = list(data.get("components", []))
    expert_xy_by_ref: Dict[str, Tuple[float, float]] = {}
    for c0 in components:
        ex0 = c0.get("expert") or {}
        if "xy_mm" in ex0:
            expert_xy_by_ref[str(c0.get("ref", ""))] = (float(ex0["xy_mm"][0]), float(ex0["xy_mm"][1]))
    out: Dict[str, Dict[str, Any]] = {}
    for comp in components:
        expert = comp.get("expert") or {}
        ref = str(comp["ref"])
        rot = int(expert.get("rot", 0)) if isinstance(expert, dict) else 0
        size_mm = tuple(comp.get("size_mm", [1.0, 1.0]))
        expert_heatmap0 = _extract_expert_heatmap(bbox, comp, grid_x, grid_y, sigma_cells)
        if isinstance(expert, dict) and "xy_mm" in expert and expert.get("xy_mm") is not None:
            x, y = float(expert["xy_mm"][0]), float(expert["xy_mm"][1])
        elif expert_heatmap0 is not None:
            x, y = _heatmap_peak_xy(bbox, expert_heatmap0, grid_x, grid_y)
        else:
            continue
        bb = _bbox_from_pose(x, y, size_mm=size_mm, rot_deg=rot)
        region_type = comp.get("region_type") or (comp.get("semantic") or {}).get("region_type") or "free"
        semantic_class = comp.get("semantic_class") or (comp.get("semantic") or {}).get("semantic_class") or "other"
        side_preference = comp.get("side_preference") or (comp.get("semantic") or {}).get("side_preference") or "free"
        subzone = comp.get("subzone") or (comp.get("semantic") or {}).get("subzone") or "free"
        same_side_group = comp.get("same_side_group") or (comp.get("semantic") or {}).get("same_side_group") or None
        anchor_ref = (
            comp.get("anchor_ref")
            or (comp.get("semantic") or {}).get("anchor_ref")
            or comp.get("module_anchor_ref")
            or (comp.get("module") or {}).get("anchor_ref")
            or None
        )
        if str(anchor_ref or "").strip() == ref:
            anchor_ref = None
        if str(subzone or "free").strip().lower() in {"", "free"} and anchor_ref and str(anchor_ref) in expert_xy_by_ref:
            ax, ay = expert_xy_by_ref[str(anchor_ref)]
            subzone = _infer_subzone_from_xy(x, y, ax, ay)
        critical_neighbors = tuple(comp.get("critical_neighbors") or (comp.get("semantic") or {}).get("critical_neighbors") or [])
        functional_group = comp.get("functional_group") or (comp.get("semantic") or {}).get("functional_group") or ""
        if str(functional_group or "").strip().lower() in {"", "misc", "other", "free", "none", "null"}:
            module_id = comp.get("module_id") or (comp.get("module") or {}).get("module_id") or ""
            functional_group = f"module:{module_id}" if module_id else "misc"
        review = comp.get("semantic_review") or {}
        review_status = str(review.get("review_status") or "seeded")
        auto_confidence = float(review.get("auto_confidence", 1.0) if review.get("auto_confidence", None) not in (None, "") else 1.0)
        needs_review = bool(review.get("needs_review", False))
        review_weight = semantic_review_weight(review_status, auto_confidence, needs_review)
        # Fallback keeps compatibility with old data lacking step-1 semantic labels.
        if str(region_type).strip().lower() not in _REGION_TYPE_TO_IDX:
            region_type = REGION_TYPE_NAMES[classify_region_type_from_bbox(bbox, bb, legacy_zone_edge_ratio, legacy_zone_core_ratio)]
        expert_heatmap = expert_heatmap0
        prior_heatmap = _extract_prior_heatmap(bbox, comp, grid_x, grid_y, sigma_cells)
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
            # Split supervision weights: semantic_review gates semantic labels only.
            # Expert heatmap, teacher distillation, metric shaping, and prior
            # consistency are independent streams by default.
            "review_weight": review_weight,  # legacy alias, semantic-only in train.py
            "semantic_supervision_weight": review_weight,
            "expert_supervision_weight": 1.0,
            "teacher_supervision_weight": 1.0,
            "metric_supervision_weight": 1.0,
            "prior_supervision_weight": 1.0,
            "expert_region_heatmap": expert_heatmap,
            "prior_region_heatmap": prior_heatmap,
        }
    return out


def build_action_region_indices(
    env: Any,
    ref: str,
    grid_x: int,
    grid_y: int,
    legacy_zone_edge_ratio: float,
    legacy_zone_core_ratio: float,
) -> np.ndarray:
    del ref, legacy_zone_edge_ratio, legacy_zone_core_ratio
    xmin, ymin, xmax, ymax = env.task.bbox_mm
    w_cells, h_cells = env.grid_shape()
    grid_mm = float(env.task.grid_mm)
    R = len(env.rotations)

    ix = np.arange(w_cells, dtype=np.float32)[:, None]
    iy = np.arange(h_cells, dtype=np.float32)[None, :]
    X = xmin + (ix + 0.5) * grid_mm
    Y = ymin + (iy + 0.5) * grid_mm
    X = np.broadcast_to(X, (w_cells, h_cells)).astype(np.float32)
    Y = np.broadcast_to(Y, (w_cells, h_cells)).astype(np.float32)
    base = np.zeros((w_cells, h_cells), dtype=np.int64)
    for i in range(w_cells):
        for j in range(h_cells):
            base[i, j] = _xy_to_grid_index((xmin, ymin, xmax, ymax), float(X[i, j]), float(Y[i, j]), int(grid_x), int(grid_y))
    idx = np.repeat(base.reshape(1, w_cells, h_cells), R, axis=0)
    return idx.reshape(-1)


def action_region_prior_from_predictions(
    region_heatmap_logits: torch.Tensor,
    action_heatmap_cell_idx_flat: np.ndarray,
    heatmap_action_prior_weight: float,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for runtime heatmap action scoring.")
    region_logp = torch.log_softmax(region_heatmap_logits.detach(), dim=-1).cpu().numpy().reshape(-1)
    prior = float(heatmap_action_prior_weight) * region_logp[np.asarray(action_heatmap_cell_idx_flat, dtype=np.int64)]
    prior = prior - float(np.max(prior))
    return prior.astype(np.float32)
