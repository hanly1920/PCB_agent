from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import statistics

from .region_prior import classify_region_type_from_bbox
from .utils import hpwl_from_pins, drop_mounting_holes_from_task_json


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return drop_mounting_holes_from_task_json(data) if isinstance(data, dict) else data


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


def _bbox_union(bbs: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not bbs:
        return None
    return (
        min(bb[0] for bb in bbs),
        min(bb[1] for bb in bbs),
        max(bb[2] for bb in bbs),
        max(bb[3] for bb in bbs),
    )


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def _bbox_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    dx = max(0.0, float(a[0]) - float(b[2]), float(b[0]) - float(a[2]))
    dy = max(0.0, float(a[1]) - float(b[3]), float(b[1]) - float(a[3]))
    return float(math.hypot(dx, dy))


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
    layout = comp.get("layout") or {}
    if isinstance(layout, dict) and layout.get(key) not in (None, ""):
        return layout.get(key)
    return default


def _module_id(comp: Dict[str, Any]) -> Optional[str]:
    mid = _component_semantic(comp, "module_id", None)
    if mid in (None, "", "none", "null"):
        return None
    return str(mid)


def _component_anchor_ref(comp: Dict[str, Any]) -> Optional[str]:
    ref = str(comp.get("ref", ""))
    for key in ("anchor_ref", "module_anchor_ref"):
        value = _component_semantic(comp, key, None)
        if value not in (None, "", "none", "null"):
            txt = str(value)
            if txt != ref:
                return txt
    nested = comp.get("module") or {}
    value = nested.get("anchor_ref")
    if value not in (None, "", "none", "null"):
        txt = str(value)
        if txt != ref:
            return txt
    return None


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


def _hpwl(task: Dict[str, Any], placed: Dict[str, Tuple[float, float, int]]) -> float:
    pins = _pin_positions(task, placed)
    hpwl = 0.0
    for pts in pins.values():
        if len(pts) > 1:
            hpwl += hpwl_from_pins(pts)
    return float(hpwl)


def _critical_pair_specs(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    ref_to_comp = {str(c["ref"]): c for c in task.get("components", [])}
    pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _add(a: str, b: str, *, weight: float = 1.0, reason: str = "explicit", source: str = "critical_neighbors") -> None:
        if a == b or a not in ref_to_comp or b not in ref_to_comp:
            return
        key = tuple(sorted((str(a), str(b))))
        cur = pair_map.get(key)
        try:
            w = float(weight)
        except Exception:
            w = 1.0
        if cur is None or w > float(cur.get("weight", 0.0)):
            pair_map[key] = {
                "a": key[0],
                "b": key[1],
                "weight": float(max(0.0, w)),
                "reason": str(reason or "explicit").lower(),
                "source": str(source or "critical_neighbors"),
            }

    net_to_refs: Dict[str, set[str]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        explicit_nets = _component_semantic(comp, "critical_nets", []) or []
        for net in explicit_nets:
            if net:
                net_to_refs.setdefault(str(net), set()).add(ref)

    for comp in task.get("components", []):
        ref = str(comp["ref"])
        for idx, nb in enumerate(_component_semantic(comp, "critical_neighbors", []) or []):
            if isinstance(nb, dict):
                target = str(nb.get("ref") or nb.get("neighbor") or nb.get("neighbor_ref") or "")
                reason = str(nb.get("reason") or nb.get("type") or nb.get("relation") or "explicit")
                weight = nb.get("weight", max(0.35, 1.0 - 0.08 * idx))
            else:
                target = str(nb)
                reason = "explicit"
                weight = max(0.35, 1.0 - 0.08 * idx)
            _add(ref, target, weight=weight, reason=reason, source="critical_neighbors")

    for net, refs in net_to_refs.items():
        ordered = sorted(refs)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                _add(ordered[i], ordered[j], weight=0.7, reason=f"net:{net}", source="critical_nets")

    return sorted(pair_map.values(), key=lambda d: (d["a"], d["b"], d.get("reason", "")))


def _critical_pairs(task: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(d["a"], d["b"]) for d in _critical_pair_specs(task)]


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
        if g in (None, "", "misc", "other", "free", "none", "null"):
            mid = _module_id(comp)
            if mid:
                g = f"module:{mid}"
        if g in (None, "", "misc", "other", "free", "none", "null"):
            continue
        out.setdefault(str(g), []).append(ref)
    return {k: v for k, v in out.items() if len(v) >= 2}


def _same_side_groups(task: Dict[str, Any]) -> Dict[str, List[Tuple[str, Optional[int]]]]:
    out: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        g = _component_semantic(comp, "same_side_group", None)
        if g in (None, "", "none", "null"):
            continue
        order = _component_semantic(comp, "boundary_order", None)
        try:
            order_i = None if order in (None, "") else int(order)
        except Exception:
            order_i = None
        out.setdefault(str(g), []).append((ref, order_i))
    return {k: v for k, v in out.items() if len(v) >= 2}


def _engineering_pitch_groups(task: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        g = _component_semantic(comp, "pitch_group", None) or _component_semantic(comp, "row_group", None)
        if g in (None, "", "none", "null"):
            # Backward-compatible fallback: same-side and align groups also define a candidate row.
            g = _component_semantic(comp, "same_side_group", None) or _component_semantic(comp, "align_group", None)
        if g in (None, "", "none", "null"):
            continue
        out.setdefault(str(g), []).append(ref)
    return {k: v for k, v in out.items() if len(v) >= 3}


def _module_groups(task: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for comp in task.get("components", []):
        mid = _module_id(comp)
        if mid:
            out.setdefault(str(mid), []).append(str(comp["ref"]))
    return {k: v for k, v in out.items() if len(v) >= 2}


def _rot_group_consistency(refs: List[str], placed: Dict[str, Tuple[float, float, int]]) -> Optional[float]:
    rots = [_normalize_rot(placed[r][2]) for r in refs if r in placed]
    if len(rots) < 2:
        return None
    counts: Dict[int, int] = {}
    for r in rots:
        counts[r] = counts.get(r, 0) + 1
    return float(max(counts.values()) / len(rots))


def _axis_cv(values: List[float]) -> Optional[float]:
    if len(values) < 3:
        return None
    xs = sorted(float(v) for v in values)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if len(gaps) < 2:
        return None
    mean_gap = sum(gaps) / len(gaps)
    if abs(mean_gap) < 1e-9:
        return None
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    return float(math.sqrt(var) / abs(mean_gap))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(100.0, float(q))) / 100.0 * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _actual_edge_side(board_bbox: Tuple[float, float, float, float], xy: Tuple[float, float]) -> str:
    xmin, ymin, xmax, ymax = board_bbox
    x, y = xy
    d = {
        "left": abs(float(x) - float(xmin)),
        "right": abs(float(xmax) - float(x)),
        "bottom": abs(float(y) - float(ymin)),
        "top": abs(float(ymax) - float(y)),
    }
    return min(d.items(), key=lambda kv: kv[1])[0]


def _target_or_ref_side(task: Dict[str, Any], comp: Dict[str, Any], placed: Dict[str, Tuple[float, float, int]]) -> Optional[str]:
    for key in ("side_preference", "region_type"):
        val = str(_component_semantic(comp, key, "free") or "free")
        if val in {"edge_left", "edge_right", "edge_top", "edge_bottom"}:
            return val.split("_", 1)[1]
        if val in {"left", "right", "top", "bottom"}:
            return val
    ref = str(comp["ref"])
    if ref in placed:
        return _actual_edge_side(tuple(task["board"]["bbox_mm"]), (placed[ref][0], placed[ref][1]))
    return None


def _is_connector_like(comp: Dict[str, Any]) -> bool:
    ref = str(comp.get("ref", "")).upper()
    sc = str(_component_semantic(comp, "semantic_class", "") or "").lower()
    rt = str(_component_semantic(comp, "region_type", "") or "").lower()
    role = str(_component_semantic(comp, "placement_role", "") or "").lower()
    fp = str(comp.get("footprint", comp.get("footprint_name", "")) or "").lower()
    if role in {"edge_anchor", "connector_anchor"}:
        return True
    if rt.startswith("edge_") and any(tok in sc for tok in ("interface", "connector", "io", "button", "switch", "led")):
        return True
    if ref.startswith(("J", "P", "CON", "CN", "SW", "S", "BTN", "K", "LED", "D")):
        return True
    return any(tok in fp for tok in ("connector", "header", "usb", "jack", "switch", "button", "terminal", "pinheader"))


def _is_large_or_anchor(comp: Dict[str, Any], area_threshold: float) -> bool:
    role = str(_component_semantic(comp, "placement_role", "") or "").lower()
    module_role = str(_component_semantic(comp, "module_role", "") or "").lower()
    if role in {"anchor_large", "edge_anchor", "main_anchor", "connector_anchor"}:
        return True
    if module_role in {"anchor", "root", "main", "main_anchor"}:
        return True
    size = comp.get("size_mm", [1.0, 1.0])
    area = float(size[0]) * float(size[1])
    if area >= area_threshold:
        return True
    return _is_connector_like(comp)


def evaluate_task_semantic_metrics(
    task: Dict[str, Any],
    placed: Dict[str, Tuple[float, float, int]],
    *,
    reference_placed: Optional[Dict[str, Tuple[float, float, int]]] = None,
    legacy_zone_edge_ratio: float = 0.12,
    legacy_zone_core_ratio: float = 0.28,
) -> Dict[str, Any]:
    xmin, ymin, xmax, ymax = task["board"]["bbox_mm"]
    board_bbox = (float(xmin), float(ymin), float(xmax), float(ymax))
    board_w = max(1e-6, float(xmax) - float(xmin))
    board_h = max(1e-6, float(ymax) - float(ymin))
    board_diag = math.hypot(board_w, board_h)
    grid = float(task.get("board", {}).get("grid_mm", 1.0))

    if reference_placed is None:
        reference_placed = _collect_expert_placements(task)

    ref_to_comp = {str(c["ref"]): c for c in task.get("components", [])}
    centers: Dict[str, Tuple[float, float]] = {}
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
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
        bboxes[ref] = bb
        idx = classify_region_type_from_bbox((xmin, ymin, xmax, ymax), bb, legacy_zone_edge_ratio, legacy_zone_core_ratio)
        actual_region[ref] = region_names[int(idx)]

    ref_centers = {r: (p[0], p[1]) for r, p in reference_placed.items()}
    ref_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for r, pose in reference_placed.items():
        comp = ref_to_comp.get(r)
        if comp is None:
            continue
        ref_bboxes[r] = _bbox_from_pose(pose[0], pose[1], tuple(comp.get("size_mm", [1.0, 1.0])), pose[2])

    edge_targets = []
    edge_hits = 0
    core_targets = []
    core_hits = 0
    edge_non_interface_count = 0
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
        sc = str(_component_semantic(comp, "semantic_class", "") or "").lower()
        side_pref = str(_component_semantic(comp, "side_preference", "free") or "free")
        role = str(_component_semantic(comp, "placement_role", "") or "").lower()
        is_interface = (
            target_region.startswith("edge_")
            or side_pref.startswith("edge_")
            or role in {"edge_anchor", "connector_anchor"}
            or any(tok in sc for tok in ("interface", "connector", "io", "button", "switch", "led"))
            or _is_connector_like(comp)
        )
        if actual_region[ref].startswith("edge_") and not is_interface:
            edge_non_interface_count += 1

    critical_specs = _critical_pair_specs(task)
    critical_distances = []
    critical_weighted_distances = []
    critical_reason_distances: Dict[str, List[float]] = {}
    for spec in critical_specs:
        a, b = str(spec.get("a")), str(spec.get("b"))
        if a in centers and b in centers:
            xa, ya = centers[a]
            xb, yb = centers[b]
            d = (abs(xa - xb) + abs(ya - yb)) / board_diag
            critical_distances.append(d)
            w = float(spec.get("weight", 1.0) or 1.0)
            critical_weighted_distances.append(d * max(0.0, w))
            reason = str(spec.get("reason", "explicit") or "explicit").lower()
            critical_reason_distances.setdefault(reason, []).append(d)

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

    # Engineering neatness metrics: orientation consistency, equal-pitch quality,
    # module overlap/separation, and boundary row regularity.
    orientation_scores: List[float] = []
    orient_sources = list(_align_groups(task).values()) + [[r for r, _ in v] for v in _same_side_groups(task).values()]
    seen_orient = set()
    for refs in orient_sources:
        key = tuple(sorted(refs))
        if key in seen_orient:
            continue
        seen_orient.add(key)
        val = _rot_group_consistency(refs, placed)
        if val is not None:
            orientation_scores.append(val)

    pitch_cvs: List[float] = []
    for refs in _engineering_pitch_groups(task).values():
        pts = [centers[r] for r in refs if r in centers]
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        axis_vals = xs if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else ys
        cv = _axis_cv(axis_vals)
        if cv is not None:
            pitch_cvs.append(cv)

    boundary_pitch_errors: List[float] = []
    for entries in _same_side_groups(task).values():
        refs = [r for r, _ in entries if r in centers]
        if len(refs) < 3:
            continue
        side_votes = []
        for r in refs:
            comp = ref_to_comp.get(r)
            side_votes.append(_target_or_ref_side(task, comp, placed) if comp is not None else None)
        side_votes = [s for s in side_votes if s in {"left", "right", "top", "bottom"}]
        if side_votes:
            side = max(sorted(set(side_votes)), key=side_votes.count)
            vals = [centers[r][1] if side in {"left", "right"} else centers[r][0] for r in refs]
        else:
            xs = [centers[r][0] for r in refs]
            ys = [centers[r][1] for r in refs]
            vals = ys if (max(xs) - min(xs)) <= (max(ys) - min(ys)) else xs
        cv = _axis_cv(vals)
        if cv is not None:
            boundary_pitch_errors.append(cv)

    module_overlap_ratios: List[float] = []
    module_bboxes_for_pred: Dict[str, Tuple[float, float, float, float]] = {}
    for mid, refs in _module_groups(task).items():
        bbs = [bboxes[r] for r in refs if r in bboxes]
        if len(bbs) >= 2:
            ub = _bbox_union(bbs)
            if ub is not None:
                module_bboxes_for_pred[mid] = ub
    module_ids = sorted(module_bboxes_for_pred)
    for i, a in enumerate(module_ids):
        for b in module_ids[i + 1:]:
            module_overlap_ratios.append(_bbox_iou(module_bboxes_for_pred[a], module_bboxes_for_pred[b]))

    # Pairwise gap / local density / corner mass: these catch collapse that HPWL may hide.
    nearest_gaps: List[float] = []
    density_scores: List[float] = []
    refs = list(centers.keys())
    radius = max(3.0 * grid, min(8.0, 0.08 * board_diag))
    for i, a in enumerate(refs):
        gaps = []
        neigh = 0
        ax, ay = centers[a]
        for j, b in enumerate(refs):
            if i == j:
                continue
            gaps.append(_bbox_gap(bboxes[a], bboxes[b]))
            bx, by = centers[b]
            if math.hypot(ax - bx, ay - by) <= radius:
                neigh += 1
        if gaps:
            nearest_gaps.append(min(gaps))
        density_scores.append(float(neigh))
    min_gap_p10 = _percentile(nearest_gaps, 10)
    min_gap_p25 = _percentile(nearest_gaps, 25)
    local_density_p90 = _percentile(density_scores, 90)

    corner_margin_x = 0.25 * board_w
    corner_margin_y = 0.25 * board_h
    corner_count = 0
    for x, y in centers.values():
        near_x = (x <= xmin + corner_margin_x) or (x >= xmax - corner_margin_x)
        near_y = (y <= ymin + corner_margin_y) or (y >= ymax - corner_margin_y)
        if near_x and near_y:
            corner_count += 1
    corner_mass_ratio = corner_count / max(1, len(centers))

    # Module centroid and bbox IoU versus original/expert.
    module_refs: Dict[str, List[str]] = {}
    for comp in task.get("components", []):
        mid = _module_id(comp)
        if mid:
            module_refs.setdefault(mid, []).append(str(comp["ref"]))
    module_centroid_errors: List[float] = []
    module_ious: List[float] = []
    for _mid, mrefs in module_refs.items():
        pred_pts = [centers[r] for r in mrefs if r in centers]
        ref_pts = [ref_centers[r] for r in mrefs if r in ref_centers]
        if len(pred_pts) >= 1 and len(ref_pts) >= 1:
            pcx = sum(x for x, _ in pred_pts) / len(pred_pts)
            pcy = sum(y for _, y in pred_pts) / len(pred_pts)
            rcx = sum(x for x, _ in ref_pts) / len(ref_pts)
            rcy = sum(y for _, y in ref_pts) / len(ref_pts)
            module_centroid_errors.append(math.hypot(pcx - rcx, pcy - rcy) / board_diag)
        pbb = _bbox_union([bboxes[r] for r in mrefs if r in bboxes])
        rbb = _bbox_union([ref_bboxes[r] for r in mrefs if r in ref_bboxes])
        if pbb is not None and rbb is not None:
            module_ious.append(_bbox_iou(pbb, rbb))

    # Anchor distance error: preserve each member-anchor relative distance.
    anchor_errors: List[float] = []
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        anchor = _component_anchor_ref(comp)
        if not anchor:
            continue
        if ref in centers and anchor in centers and ref in ref_centers and anchor in ref_centers:
            pd = math.hypot(centers[ref][0] - centers[anchor][0], centers[ref][1] - centers[anchor][1])
            rd = math.hypot(ref_centers[ref][0] - ref_centers[anchor][0], ref_centers[ref][1] - ref_centers[anchor][1])
            anchor_errors.append(abs(pd - rd) / board_diag)

    areas = []
    for comp in task.get("components", []):
        size = comp.get("size_mm", [1.0, 1.0])
        areas.append(float(size[0]) * float(size[1]))
    area_thr = _percentile(areas, 85) or 0.0
    large_drifts: List[float] = []
    for comp in task.get("components", []):
        ref = str(comp["ref"])
        if not _is_large_or_anchor(comp, area_thr):
            continue
        if ref in centers and ref in ref_centers:
            large_drifts.append(math.hypot(centers[ref][0] - ref_centers[ref][0], centers[ref][1] - ref_centers[ref][1]))

    # Connector side accuracy versus original/expert side.
    connector_hits = 0
    connector_total = 0
    for comp in task.get("components", []):
        if not _is_connector_like(comp):
            continue
        ref = str(comp["ref"])
        if ref not in centers or ref not in ref_centers:
            continue
        pred_side = _actual_edge_side(board_bbox, centers[ref])
        ref_side = _target_or_ref_side(task, comp, reference_placed)
        if ref_side in {"left", "right", "top", "bottom"}:
            connector_total += 1
            if pred_side == ref_side:
                connector_hits += 1

    # Same-side order accuracy: pairwise order agreement inside each tagged edge group.
    order_hits = 0
    order_total = 0
    for _g, items in _same_side_groups(task).items():
        refs_with_order = [(r, o) for r, o in items if o is not None and r in centers]
        if len(refs_with_order) < 2:
            continue
        # Use the preferred/ref side to choose the ordering axis.
        side_votes: List[str] = []
        for r, _o in refs_with_order:
            comp = ref_to_comp.get(r)
            if comp is None:
                continue
            side = _target_or_ref_side(task, comp, reference_placed)
            if side in {"left", "right", "top", "bottom"}:
                side_votes.append(side)
        side = max(side_votes, key=side_votes.count) if side_votes else None
        axis = "y" if side in {"left", "right"} else "x"
        for i in range(len(refs_with_order)):
            for j in range(i + 1, len(refs_with_order)):
                ri, oi = refs_with_order[i]
                rj, oj = refs_with_order[j]
                if oi == oj:
                    continue
                vi = centers[ri][1] if axis == "y" else centers[ri][0]
                vj = centers[rj][1] if axis == "y" else centers[rj][0]
                order_total += 1
                if (oi < oj and vi <= vj) or (oi > oj and vi >= vj):
                    order_hits += 1

    hpwl = _hpwl(task, placed)
    ref_hpwl = _hpwl(task, reference_placed) if reference_placed else None
    hpwl_ratio = (hpwl / ref_hpwl) if ref_hpwl and ref_hpwl > 1e-9 else None

    large_drift_max = max(large_drifts) if large_drifts else None
    large_drift_mean = _mean(large_drifts)

    return {
        "n_components": len(task.get("components", [])),
        "n_placed": len(centers),
        "hpwl": float(hpwl),
        "reference_hpwl": float(ref_hpwl) if ref_hpwl is not None else None,
        "hpwl_ratio": float(hpwl_ratio) if hpwl_ratio is not None else None,

        "min_gap_p10": float(min_gap_p10) if min_gap_p10 is not None else None,
        "min_gap_p25": float(min_gap_p25) if min_gap_p25 is not None else None,
        "local_density_p90": float(local_density_p90) if local_density_p90 is not None else None,
        "corner_mass_ratio": float(corner_mass_ratio),
        "edge_non_interface_count": int(edge_non_interface_count),

        "module_centroid_error": float(_mean(module_centroid_errors)) if module_centroid_errors else None,
        "module_centroid_error_max": float(max(module_centroid_errors)) if module_centroid_errors else None,
        "module_bbox_iou_with_original": float(_mean(module_ious)) if module_ious else None,
        "anchor_distance_error": float(_mean(anchor_errors)) if anchor_errors else None,
        "anchor_distance_error_max": float(max(anchor_errors)) if anchor_errors else None,
        "large_component_drift_mm": float(large_drift_max) if large_drift_max is not None else None,
        "large_component_drift_mean_mm": float(large_drift_mean) if large_drift_mean is not None else None,
        "large_component_drift_ratio": float(large_drift_max / board_diag) if large_drift_max is not None else None,
        "connector_side_accuracy": (connector_hits / connector_total) if connector_total else None,
        "connector_side_count": int(connector_total),
        "same_side_order_accuracy": (order_hits / order_total) if order_total else None,
        "same_side_order_pair_count": int(order_total),

        "edge_hit_rate": (edge_hits / len(edge_targets)) if edge_targets else None,
        "edge_target_count": len(edge_targets),
        "core_center_rate": (core_hits / len(core_targets)) if core_targets else None,
        "core_target_count": len(core_targets),
        "critical_connection_avg_distance": (sum(critical_distances) / len(critical_distances)) if critical_distances else None,
        "critical_connection_weighted_avg_distance": (sum(critical_weighted_distances) / max(1e-9, sum(max(0.0, float(s.get("weight", 1.0) or 1.0)) for s in critical_specs if s.get("a") in centers and s.get("b") in centers))) if critical_weighted_distances else None,
        "critical_pair_count": len(critical_distances),
        "decoupling_distance_p90": _percentile(critical_reason_distances.get("decoupling", []) + critical_reason_distances.get("bypass", []), 90),
        "clock_load_distance_p90": _percentile(critical_reason_distances.get("clock", []) + critical_reason_distances.get("crystal_load", []), 90),
        "esd_to_connector_distance_p90": _percentile(critical_reason_distances.get("esd", []) + critical_reason_distances.get("connector_protection", []), 90),
        "power_loop_distance_p90": _percentile(critical_reason_distances.get("power_loop", []) + critical_reason_distances.get("hot_loop", []) + critical_reason_distances.get("switching_loop", []), 90),
        "align_group_avg_deviation": (sum(align_group_scores) / len(align_group_scores)) if align_group_scores else None,
        "align_group_count": len(align_group_scores),
        "orientation_consistency": (sum(orientation_scores) / len(orientation_scores)) if orientation_scores else None,
        "orientation_group_count": len(orientation_scores),
        "pitch_cv": (sum(pitch_cvs) / len(pitch_cvs)) if pitch_cvs else None,
        "pitch_group_count": len(pitch_cvs),
        "boundary_pitch_error": (sum(boundary_pitch_errors) / len(boundary_pitch_errors)) if boundary_pitch_errors else None,
        "boundary_pitch_group_count": len(boundary_pitch_errors),
        "module_overlap_ratio": (sum(module_overlap_ratios) / len(module_overlap_ratios)) if module_overlap_ratios else None,
        "module_overlap_pair_count": len(module_overlap_ratios),
        "functional_group_compactness": (sum(compactness_scores) / len(compactness_scores)) if compactness_scores else None,
        "functional_group_count": len(compactness_scores),
    }


def _mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def evaluate_task_file(
    task_path: str | Path,
    pred_path: Optional[str | Path] = None,
    *,
    legacy_zone_edge_ratio: float = 0.12,
    legacy_zone_core_ratio: float = 0.28,
) -> Dict[str, Any]:
    task = _load_json(task_path)
    reference = _collect_expert_placements(task)
    if pred_path is None:
        placed = reference
    else:
        pred = _load_json(pred_path)
        placed = _collect_pred_placements(pred)
    m = evaluate_task_semantic_metrics(
        task,
        placed,
        reference_placed=reference,
        legacy_zone_edge_ratio=legacy_zone_edge_ratio,
        legacy_zone_core_ratio=legacy_zone_core_ratio,
    )
    m["task"] = Path(task_path).stem
    if pred_path is not None:
        m["prediction"] = str(pred_path)
    return m


def summarize_semantic_metrics(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "hpwl", "reference_hpwl", "hpwl_ratio",
        "min_gap_p10", "min_gap_p25", "local_density_p90", "corner_mass_ratio",
        "edge_non_interface_count", "module_centroid_error", "module_centroid_error_max",
        "module_bbox_iou_with_original", "anchor_distance_error", "anchor_distance_error_max",
        "large_component_drift_mm", "large_component_drift_mean_mm", "large_component_drift_ratio",
        "connector_side_accuracy", "same_side_order_accuracy",
        "edge_hit_rate", "core_center_rate", "critical_connection_avg_distance",
        "critical_connection_weighted_avg_distance", "decoupling_distance_p90",
        "clock_load_distance_p90", "esd_to_connector_distance_p90", "power_loop_distance_p90",
        "align_group_avg_deviation", "orientation_consistency", "pitch_cv",
        "boundary_pitch_error", "module_overlap_ratio", "functional_group_compactness",
    ]
    summary: Dict[str, Any] = {
        "n_tasks": len(per_task),
        "n_components_total": int(sum(int(m.get("n_components", 0)) for m in per_task)),
        "n_placed_total": int(sum(int(m.get("n_placed", 0)) for m in per_task)),
    }
    for key in keys:
        summary[f"{key}_mean"] = _mean_ignore_none(m.get(key) for m in per_task)
        vals = [float(m[key]) for m in per_task if m.get(key) is not None]
        if vals:
            summary[f"{key}_p50"] = _percentile(vals, 50)
            summary[f"{key}_p90"] = _percentile(vals, 90)
            if key in {"hpwl_ratio", "corner_mass_ratio", "large_component_drift_ratio"}:
                if key == "corner_mass_ratio":
                    summary["corner_collapse_count_gt_0p35"] = int(sum(1 for v in vals if v > 0.35))
                elif key == "large_component_drift_ratio":
                    summary["large_drift_count_gt_0p15diag"] = int(sum(1 for v in vals if v > 0.15))
                elif key == "hpwl_ratio":
                    summary["hpwl_worse_count_gt_1p0"] = int(sum(1 for v in vals if v > 1.0))
                    summary["hpwl_bad_count_gt_1p5"] = int(sum(1 for v in vals if v > 1.5))
    return summary
