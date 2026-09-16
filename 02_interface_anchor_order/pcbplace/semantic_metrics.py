from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import statistics

from .region_prior import classify_region_type_from_bbox
from .utils import hpwl_from_pins


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_rot(rot: Any) -> int:
    return int(round(float(rot))) % 360


def _swap_size_if_needed(size_mm: Tuple[float, float], rot_deg: int) -> Tuple[float, float]:
    w, h = float(size_mm[0]), float(size_mm[1])
    if _normalize_rot(rot_deg) % 180 != 0:
        return h, w
    return w, h


def _bbox_from_pose(x: float, y: float, size_mm: Tuple[float, float], rot_deg: int) -> Tuple[float, float, float, float]:
    w, h = _swap_size_if_needed(size_mm, rot_deg)
    return (float(x) - 0.5 * w, float(y) - 0.5 * h, float(x) + 0.5 * w, float(y) + 0.5 * h)


def _rotate_rel(rel_xy_mm: Tuple[float, float], rot_deg: int) -> Tuple[float, float]:
    x, y = float(rel_xy_mm[0]), float(rel_xy_mm[1])
    rot = _normalize_rot(rot_deg)
    if rot == 0:
        return x, y
    if rot == 90:
        return -y, x
    if rot == 180:
        return -x, -y
    if rot == 270:
        return y, -x
    th = math.radians(float(rot))
    c = math.cos(th)
    s = math.sin(th)
    return (x * c - y * s, x * s + y * c)


def _component_semantic(comp: Dict[str, Any], key: str, default: Any) -> Any:
    if key in comp and comp.get(key) not in (None, ""):
        return comp.get(key)
    nested = comp.get("semantic") or {}
    if nested.get(key) not in (None, ""):
        return nested.get(key)
    return default


def _collect_expert_placements(task: Dict[str, Any]) -> Dict[str, Tuple[float, float, int]]:
    placed: Dict[str, Tuple[float, float, int]] = {}
    for comp in task.get("components", []):
        ex = comp.get("expert") or {}
        if "xy_mm" in ex:
            xy = ex["xy_mm"]
            placed[str(comp["ref"])] = (float(xy[0]), float(xy[1]), _normalize_rot(ex.get("rot", 0)))
    return placed


def _collect_pred_placements(pred: Dict[str, Any]) -> Dict[str, Tuple[float, float, int]]:
    placed: Dict[str, Tuple[float, float, int]] = {}
    for ref, pose in (pred.get("placed") or {}).items():
        if isinstance(pose, dict):
            x = pose.get("x_mm", pose.get("x", 0.0))
            y = pose.get("y_mm", pose.get("y", 0.0))
            rot = pose.get("rot", 0)
        else:
            x, y, rot = pose
        placed[str(ref)] = (float(x), float(y), _normalize_rot(rot))
    return placed


def _pin_positions(task: Dict[str, Any], placed: Dict[str, Tuple[float, float, int]]) -> Dict[str, List[Tuple[float, float]]]:
    pins: Dict[str, List[Tuple[float, float]]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        if ref not in placed:
            continue
        x0, y0, rot = placed[ref]
        for pad in comp.get("pads", []):
            if isinstance(pad, dict):
                net = str(pad.get("net", "") or "")
                rel = tuple(pad.get("rel_mm", [0.0, 0.0]))
            else:
                net, rel = pad
            rx, ry = _rotate_rel((float(rel[0]), float(rel[1])), rot)
            pins.setdefault(net, []).append((x0 + rx, y0 + ry))
    return pins


def _critical_pairs(task: Dict[str, Any]) -> List[Tuple[str, str]]:
    ref_to_comp = {str(c["ref"]): c for c in task.get("components", [])}
    pair_set = set()

    net_to_refs: Dict[str, set[str]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        explicit_nets = _component_semantic(comp, "critical_nets", []) or []
        for net in explicit_nets:
            if net:
                net_to_refs.setdefault(str(net), set()).add(ref)

    for comp in task.get("components", []):
        ref = str(comp["ref"])
        for nb in (_component_semantic(comp, "critical_neighbors", []) or []):
            nb = str(nb)
            if nb in ref_to_comp and nb != ref:
                pair_set.add(tuple(sorted((ref, nb))))

    for refs in net_to_refs.values():
        ordered = sorted(refs)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pair_set.add((ordered[i], ordered[j]))

    return sorted(pair_set)


def _align_groups(task: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        g = _component_semantic(comp, "align_group", None)
        if g in (None, "", "none"):
            continue
        out.setdefault(str(g), []).append(ref)
    return {k: v for k, v in out.items() if len(v) >= 2}


def _functional_groups(task: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        g = _component_semantic(comp, "functional_group", "misc")
        if g in (None, "", "misc", "other"):
            continue
        out.setdefault(str(g), []).append(ref)
    return {k: v for k, v in out.items() if len(v) >= 2}


def evaluate_task_semantic_metrics(
    task: Dict[str, Any],
    placed: Dict[str, Tuple[float, float, int]],
    *,
    zone_edge_ratio: float = 0.12,
    zone_core_ratio: float = 0.28,
) -> Dict[str, Any]:
    xmin, ymin, xmax, ymax = task["board"]["bbox_mm"]
    board_w = max(1e-6, float(xmax) - float(xmin))
    board_h = max(1e-6, float(ymax) - float(ymin))
    board_diag = math.hypot(board_w, board_h)

    ref_to_comp = {str(c["ref"]): c for c in task.get("components", [])}
    centers: Dict[str, Tuple[float, float]] = {}
    actual_region: Dict[str, str] = {}
    region_names = ["edge_top", "edge_bottom", "edge_left", "edge_right", "core", "free"]

    for ref, pose in placed.items():
        comp = ref_to_comp.get(ref)
        if comp is None:
            continue
        x, y, rot = pose
        size_mm = tuple(comp.get("size_mm", [1.0, 1.0]))
        bb = _bbox_from_pose(x, y, size_mm=size_mm, rot_deg=rot)
        centers[ref] = (float(x), float(y))
        idx = classify_region_type_from_bbox((xmin, ymin, xmax, ymax), bb, zone_edge_ratio, zone_core_ratio)
        actual_region[ref] = region_names[int(idx)]

    edge_targets = []
    edge_hits = 0
    core_targets = []
    core_hits = 0
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        target_region = str(_component_semantic(comp, "region_type", "free"))
        if ref not in actual_region:
            continue
        if target_region.startswith("edge_"):
            edge_targets.append(ref)
            if actual_region[ref] == target_region:
                edge_hits += 1
        if target_region == "core":
            core_targets.append(ref)
            if actual_region[ref] == "core":
                core_hits += 1

    critical_pairs = _critical_pairs(task)
    critical_distances = []
    for a, b in critical_pairs:
        if a in centers and b in centers:
            xa, ya = centers[a]
            xb, yb = centers[b]
            critical_distances.append((abs(xa - xb) + abs(ya - yb)) / board_diag)

    align_group_scores = []
    align_groups = _align_groups(task)
    for refs in align_groups.values():
        pts = [centers[r] for r in refs if r in centers]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        med_x = statistics.median(xs)
        med_y = statistics.median(ys)
        dev_x = sum(abs(x - med_x) for x in xs) / len(xs) / board_w
        dev_y = sum(abs(y - med_y) for y in ys) / len(ys) / board_h
        align_group_scores.append(min(dev_x, dev_y))

    compactness_scores = []
    functional_groups = _functional_groups(task)
    for refs in functional_groups.values():
        pts = [centers[r] for r in refs if r in centers]
        if len(pts) < 2:
            continue
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        compactness = sum(abs(x - cx) + abs(y - cy) for x, y in pts) / len(pts) / board_diag
        compactness_scores.append(compactness)

    pins = _pin_positions(task, placed)
    hpwl = 0.0
    for pts in pins.values():
        if len(pts) > 1:
            hpwl += hpwl_from_pins(pts)

    return {
        "n_components": len(task.get("components", [])),
        "n_placed": len(centers),
        "edge_hit_rate": (edge_hits / len(edge_targets)) if edge_targets else None,
        "edge_target_count": len(edge_targets),
        "core_center_rate": (core_hits / len(core_targets)) if core_targets else None,
        "core_target_count": len(core_targets),
        "critical_connection_avg_distance": (sum(critical_distances) / len(critical_distances)) if critical_distances else None,
        "critical_pair_count": len(critical_distances),
        "align_group_avg_deviation": (sum(align_group_scores) / len(align_group_scores)) if align_group_scores else None,
        "align_group_count": len(align_group_scores),
        "functional_group_compactness": (sum(compactness_scores) / len(compactness_scores)) if compactness_scores else None,
        "functional_group_count": len(compactness_scores),
        "hpwl": float(hpwl),
    }


def _mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def evaluate_task_file(
    task_path: str | Path,
    pred_path: str | Path | None = None,
    *,
    zone_edge_ratio: float = 0.12,
    zone_core_ratio: float = 0.28,
) -> Dict[str, Any]:
    task = _load_json(task_path)
    if pred_path is None:
        placed = _collect_expert_placements(task)
        layout_source = "expert"
    else:
        placed = _collect_pred_placements(_load_json(pred_path))
        layout_source = str(pred_path)
    metrics = evaluate_task_semantic_metrics(
        task,
        placed,
        zone_edge_ratio=zone_edge_ratio,
        zone_core_ratio=zone_core_ratio,
    )
    metrics["task"] = str(task_path)
    metrics["layout_source"] = layout_source
    return metrics


def summarize_semantic_metrics(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_tasks": len(per_task),
        "edge_hit_rate_mean": _mean_ignore_none(m.get("edge_hit_rate") for m in per_task),
        "core_center_rate_mean": _mean_ignore_none(m.get("core_center_rate") for m in per_task),
        "critical_connection_avg_distance_mean": _mean_ignore_none(m.get("critical_connection_avg_distance") for m in per_task),
        "align_group_avg_deviation_mean": _mean_ignore_none(m.get("align_group_avg_deviation") for m in per_task),
        "functional_group_compactness_mean": _mean_ignore_none(m.get("functional_group_compactness") for m in per_task),
        "hpwl_mean": _mean_ignore_none(m.get("hpwl") for m in per_task),
    }
