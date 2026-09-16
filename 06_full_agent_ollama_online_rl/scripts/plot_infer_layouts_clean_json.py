#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean PCB layout comparison plotter: Manual JSON vs Infer JSON.

Style:
  - Component rectangles are line-only. No fill color.
  - Components with the same module_id use the same border color.
  - Manual and Infer panels share the exact same module color map per board.
  - Drawing is based on JSON only; no KiCad courtyard/silk/fab parsing.

Recommended command:
  python scripts/plot_infer_layouts_clean_json.py \
    --infer_dir infer_out/spacing_clean_bs4_k108_step0051438_all_data \
    --manual_root infer_manual \
    --task_root data/infer \
    --out_dir plots/spacing_clean_bs4_k108_step0051438_manual_json \
    --pdf plots/spacing_clean_bs4_k108_step0051438_manual_json/layout_compare.pdf \
    --labels auto

Meaning of roots:
  --infer_dir:
      Model infer outputs, usually *.infer.json. Reads top-level placed/placed_raw.

  --manual_root:
      Manual/reference JSONs, now expected from infer_manual/.
      This can be either:
        1) infer-style JSON with top-level placed / placed_raw / manual_placed
        2) dataset-style JSON with components[*].expert.xy_mm
        3) dataset-style JSON with components[*].manual.xy_mm

  --task_root:
      Base task JSONs used for component size_mm, module_id and pads/nets.
      Usually data/infer. If manual_root JSON already contains components, this is optional,
      but keeping --task_root data/infer is recommended.

HPWL:
  - Prefer infer JSON metrics.reference_hpwl / metrics.hpwl / metrics.hpwl_ratio.
  - If missing, compute from task pads/nets.

If matplotlib is missing:
  python -m pip install matplotlib
or:
  conda install -c conda-forge matplotlib -y
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon, Rectangle
except ModuleNotFoundError as e:
    raise SystemExit(
        "[error] matplotlib is not installed.\n"
        "Install it first:\n"
        "  python -m pip install matplotlib\n"
        "or:\n"
        "  conda install -c conda-forge matplotlib -y\n"
    ) from e


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Placement = Dict[str, Tuple[float, float, float]]


# ----------------------------
# Basic utilities
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", s).lower()


def logical_stem(path_or_name: Any) -> str:
    name = Path(str(path_or_name)).name
    for suf in [
        ".infer.json",
        ".manual.json",
        ".manual.infer.json",
        ".placed.json",
        ".layout.json",
        ".json",
    ]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


def fmt_num(v: Any, nd: int = 2) -> str:
    try:
        if v is None:
            return "NA"
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return "NA"
        return f"{x:.{nd}f}"
    except Exception:
        return "NA"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_size(comp: Dict[str, Any]) -> Tuple[float, float]:
    size = (
        comp.get("size_mm")
        or comp.get("bbox_size_mm")
        or comp.get("courtyard_size_mm")
        or comp.get("size")
        or comp.get("bbox_mm")
    )
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return max(safe_float(size[0], 1.0), 0.05), max(safe_float(size[1], 1.0), 0.05)
    return 1.0, 1.0


def transform_point(local: Point, origin: Point, rot_deg: float) -> Point:
    x, y = local
    ox, oy = origin
    th = math.radians(rot_deg)
    co, si = math.cos(th), math.sin(th)
    return ox + x * co - y * si, oy + x * si + y * co


def rotate_point(dx: float, dy: float, rot_deg: float) -> Tuple[float, float]:
    th = math.radians(rot_deg)
    co, si = math.cos(th), math.sin(th)
    return co * dx - si * dy, si * dx + co * dy


def rect_polygon(center: Point, size: Tuple[float, float], rot_deg: float) -> List[Point]:
    w, h = size
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [transform_point(p, center, rot_deg) for p in pts]


def bbox_from_points(points: Iterable[Point]) -> Optional[BBox]:
    pts = list(points)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_union(boxes: Iterable[Optional[BBox]]) -> Optional[BBox]:
    real = [b for b in boxes if b is not None]
    if not real:
        return None
    return (
        min(b[0] for b in real),
        min(b[1] for b in real),
        max(b[2] for b in real),
        max(b[3] for b in real),
    )


# ----------------------------
# File matching
# ----------------------------

def build_json_index(root: Optional[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if root is None:
        return index
    if not root.exists():
        return index

    files = [root] if root.is_file() and root.suffix.lower() == ".json" else list(root.rglob("*.json"))

    for p in files:
        if p.name.lower() in {"manifest.json", "conversion_summary.json", "layout_plot_summary.json"}:
            continue
        key = logical_stem(p)
        old = index.get(key)
        if old is None:
            index[key] = p
        else:
            # Prefer shorter paths and infer_manual/data/infer roots.
            p_score = len(str(p))
            old_score = len(str(old))
            sp = str(p).replace("\\", "/")
            so = str(old).replace("\\", "/")
            if "/infer_manual/" in sp:
                p_score -= 1000
            if "/data/infer/" in sp:
                p_score -= 500
            if "/infer_manual/" in so:
                old_score -= 1000
            if "/data/infer/" in so:
                old_score -= 500
            if p_score < old_score:
                index[key] = p
    return index


def find_by_stem(stem: str, index: Dict[str, Path]) -> Optional[Path]:
    if stem in index:
        return index[stem]
    n = normalize_name(stem)
    for k, p in index.items():
        if normalize_name(k) == n:
            return p

    candidates = []
    for k, p in index.items():
        nk = normalize_name(k)
        if nk and (nk in n or n in nk):
            candidates.append((abs(len(k) - len(stem)), p))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None


# ----------------------------
# Placement parsing
# ----------------------------

def parse_placement_mapping(obj: Any) -> Placement:
    out: Placement = {}

    if isinstance(obj, dict):
        # Some files nest placement under {"placements": {"U1": ...}}.
        if "placements" in obj and isinstance(obj["placements"], (dict, list)):
            nested = parse_placement_mapping(obj["placements"])
            if nested:
                return nested

        for ref, v in obj.items():
            if str(ref) in {
                "metrics",
                "task",
                "components",
                "board",
                "nets",
                "placed",
                "placed_raw",
                "manual",
                "expert",
                "summary",
            }:
                continue

            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[str(ref)] = (
                    safe_float(v[0]),
                    safe_float(v[1]),
                    safe_float(v[2], 0.0) if len(v) >= 3 else 0.0,
                )
            elif isinstance(v, dict):
                xy = v.get("xy_mm") or v.get("center_mm") or v.get("pos_mm") or v.get("at_mm") or v.get("xy")
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    out[str(ref)] = (
                        safe_float(xy[0]),
                        safe_float(xy[1]),
                        safe_float(v.get("rot", v.get("rotation", v.get("rot_deg", 0.0)))),
                    )
                elif ("x_mm" in v and "y_mm" in v) or ("x" in v and "y" in v):
                    out[str(ref)] = (
                        safe_float(v.get("x_mm", v.get("x", 0.0))),
                        safe_float(v.get("y_mm", v.get("y", 0.0))),
                        safe_float(v.get("rot", v.get("rotation", v.get("rot_deg", 0.0)))),
                    )

    elif isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("ref", item.get("reference", "")))
            if not ref:
                continue
            xy = item.get("xy_mm") or item.get("center_mm") or item.get("pos_mm") or item.get("at_mm") or item.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                out[ref] = (
                    safe_float(xy[0]),
                    safe_float(xy[1]),
                    safe_float(item.get("rot", item.get("rotation", item.get("rot_deg", 0.0)))),
                )
            elif ("x_mm" in item and "y_mm" in item) or ("x" in item and "y" in item):
                out[ref] = (
                    safe_float(item.get("x_mm", item.get("x", 0.0))),
                    safe_float(item.get("y_mm", item.get("y", 0.0))),
                    safe_float(item.get("rot", item.get("rotation", item.get("rot_deg", 0.0)))),
                )

    return out


def parse_components_placement(task: Dict[str, Any], source: str = "auto") -> Placement:
    """
    Reads placement from components.

    source=auto order:
      expert -> manual -> placed -> component xy fields
    """
    out: Placement = {}

    if source == "auto":
        keys = ["expert", "manual", "placed", "placement", "component"]
    elif source in {"component", "self"}:
        keys = ["component"]
    else:
        keys = [source]

    for comp in task.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        ref = str(comp.get("ref", comp.get("reference", "")))
        if not ref:
            continue

        for key in keys:
            src = comp if key == "component" else comp.get(key)
            if not isinstance(src, dict):
                continue

            xy = (
                src.get("xy_mm")
                or src.get("center_mm")
                or src.get("pos_mm")
                or src.get("at_mm")
                or src.get("xy")
            )
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                out[ref] = (
                    safe_float(xy[0]),
                    safe_float(xy[1]),
                    safe_float(src.get("rot", src.get("rotation", src.get("rot_deg", 0.0)))),
                )
                break

            if ("x_mm" in src and "y_mm" in src) or ("x" in src and "y" in src):
                out[ref] = (
                    safe_float(src.get("x_mm", src.get("x", 0.0))),
                    safe_float(src.get("y_mm", src.get("y", 0.0))),
                    safe_float(src.get("rot", src.get("rotation", src.get("rot_deg", 0.0)))),
                )
                break

    return out


def parse_json_placement(doc: Dict[str, Any], mode: str = "auto") -> Placement:
    """
    mode options:
      auto: top-level manual/reference/placed fields first, then components expert/manual.
      placed: top-level placed only.
      placed_raw: top-level placed_raw only.
      expert/manual/component: components placement.
    """
    if mode == "placed":
        return parse_placement_mapping(doc.get("placed"))
    if mode == "placed_raw":
        return parse_placement_mapping(doc.get("placed_raw"))
    if mode in {"expert", "manual", "component", "self"}:
        return parse_components_placement(doc, mode)

    # auto: check common top-level placement fields first.
    for key in [
        "manual_placed",
        "reference_placed",
        "expert_placed",
        "manual",
        "reference",
        "placed",
        "placements",
        "placed_raw",
    ]:
        pl = parse_placement_mapping(doc.get(key))
        if pl:
            return pl

    # Some infer files contain task/components plus no placed field; fallback.
    pl = parse_components_placement(doc, "auto")
    if pl:
        return pl

    # Some files may nest the original task.
    if isinstance(doc.get("task"), dict):
        pl = parse_components_placement(doc["task"], "auto")
        if pl:
            return pl

    return {}


def choose_task_doc(manual_doc: Dict[str, Any], infer_doc: Dict[str, Any], task_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects the JSON that contains components with size_mm/module_id/pads.
    """
    for doc in [task_doc, manual_doc, infer_doc, manual_doc.get("task") if isinstance(manual_doc, dict) else None, infer_doc.get("task") if isinstance(infer_doc, dict) else None]:
        if isinstance(doc, dict) and isinstance(doc.get("components"), list) and doc.get("components"):
            return doc
    return manual_doc


# ----------------------------
# Metrics and drawing
# ----------------------------

def compute_hpwl_from_task(task: Dict[str, Any], placements: Placement) -> Optional[float]:
    netpts: Dict[str, List[Point]] = {}

    for comp in task.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        ref = str(comp.get("ref", comp.get("reference", "")))
        if ref not in placements:
            continue
        x, y, rot = placements[ref]
        for pad in comp.get("pads", []) or []:
            if not isinstance(pad, dict):
                continue
            net = pad.get("net")
            if not net:
                continue
            rel = pad.get("rel_mm") or pad.get("xy_mm") or pad.get("rel") or [0.0, 0.0]
            if not isinstance(rel, (list, tuple)) or len(rel) < 2:
                continue
            dx, dy = safe_float(rel[0]), safe_float(rel[1])
            rdx, rdy = rotate_point(dx, dy, rot)
            netpts.setdefault(str(net), []).append((x + rdx, y + rdy))

    if not netpts and isinstance(task.get("nets"), dict):
        # Fallback for nets that reference REF.PAD names.
        pad_rel: Dict[Tuple[str, str], Point] = {}
        for comp in task.get("components", []) or []:
            if not isinstance(comp, dict):
                continue
            ref = str(comp.get("ref", comp.get("reference", "")))
            for pad in comp.get("pads", []) or []:
                if not isinstance(pad, dict):
                    continue
                name = str(pad.get("name", ""))
                rel = pad.get("rel_mm") or pad.get("xy_mm") or [0.0, 0.0]
                if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                    pad_rel[(ref, name)] = (safe_float(rel[0]), safe_float(rel[1]))

        for net, members in (task.get("nets") or {}).items():
            if not isinstance(members, list):
                continue
            for member in members:
                s = str(member)
                if "." in s:
                    ref, pad_name = s.rsplit(".", 1)
                elif ":" in s:
                    ref, pad_name = s.rsplit(":", 1)
                else:
                    ref, pad_name = s, ""
                if ref not in placements:
                    continue
                x, y, rot = placements[ref]
                dx, dy = pad_rel.get((ref, pad_name), (0.0, 0.0))
                rdx, rdy = rotate_point(dx, dy, rot)
                netpts.setdefault(str(net), []).append((x + rdx, y + rdy))

    if not netpts:
        return None

    hpwl = 0.0
    used = 0
    for pts in netpts.values():
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
            used += 1
    return hpwl if used else None


def metric_float(infer: Dict[str, Any], key: str) -> Optional[float]:
    try:
        metrics = infer.get("metrics") or {}
        val = metrics.get(key)
        return None if val is None else float(val)
    except Exception:
        return None


def module_of_component(comp: Dict[str, Any]) -> str:
    for key in [
        "module_id",
        "module",
        "module_name",
        "group_id",
        "group",
        "cluster_id",
        "functional_group",
        "region_id",
    ]:
        val = comp.get(key)
        if val not in (None, "", "None"):
            return str(val)

    # Fallback to ref prefix so unmoduleized parts still get stable visual grouping.
    ref = str(comp.get("ref", comp.get("reference", "")))
    m = re.match(r"^[A-Za-z]+", ref)
    return m.group(0).upper() if m else "unassigned"


def stable_color_for_module(module: str) -> Any:
    """
    Stable, readable color independent of module order.
    Uses tab20 palette but chooses color by hash, so the same module name
    keeps the same color across manual/infer panels and across pages.
    """
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    if not module:
        return (0.05, 0.05, 0.05, 1.0)
    h = int(hashlib.md5(module.encode("utf-8")).hexdigest()[:8], 16)
    return palette[h % len(palette)]


def build_ref_maps(task: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    comp_by_ref: Dict[str, Dict[str, Any]] = {}
    ref_to_module: Dict[str, str] = {}

    for comp in task.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        ref = str(comp.get("ref", comp.get("reference", "")))
        if not ref:
            continue
        comp_by_ref[ref] = comp
        ref_to_module[ref] = module_of_component(comp)

    modules = sorted({m for m in ref_to_module.values() if m})
    colors = {m: stable_color_for_module(m) for m in modules}
    return comp_by_ref, ref_to_module, colors


def should_label(ref: str, size: Tuple[float, float], ncomp: int, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True

    w, h = size
    area = w * h
    if ncomp <= 35:
        return True
    if max(w, h) >= 5.0 or area >= 12.0:
        return True
    if re.match(r"^(U|J|P|CN|CONN|SW|M|X|Y|USB|HDMI|BT|LCD|LED|FUSE|FB|L)\d*", ref, re.I):
        return True
    return False


def placement_polygons(task: Dict[str, Any], placement: Placement) -> List[Tuple[str, List[Point], Tuple[float, float, float], Tuple[float, float]]]:
    comp_by_ref, _, _ = build_ref_maps(task)
    polys = []
    for ref, vals in placement.items():
        comp = comp_by_ref.get(ref)
        if not comp:
            continue
        x, y, rot = vals
        size = get_size(comp)
        poly = rect_polygon((x, y), size, rot)
        polys.append((ref, poly, (x, y, rot), size))
    # Draw large first; small on top.
    polys.sort(key=lambda t: t[3][0] * t[3][1], reverse=True)
    return polys


def derive_view_bbox(task: Dict[str, Any], manual_polys: list, infer_polys: list) -> Tuple[Optional[BBox], Optional[BBox]]:
    boxes: List[Optional[BBox]] = []
    board_bbox = None

    b = (task.get("board") or {}).get("bbox_mm") or task.get("board_bbox_mm") or task.get("bbox_mm")
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        board_bbox = (safe_float(b[0]), safe_float(b[1]), safe_float(b[2]), safe_float(b[3]))
        boxes.append(board_bbox)

    for _, poly, _, _ in manual_polys:
        boxes.append(bbox_from_points(poly))
    for _, poly, _, _ in infer_polys:
        boxes.append(bbox_from_points(poly))

    return board_bbox, bbox_union(boxes)


def setup_axis(ax: Any, title: str, view_bbox: Optional[BBox], show_ylabel: bool = True) -> None:
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_aspect("equal", adjustable="box")
    if view_bbox:
        x1, y1, x2, y2 = view_bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        pad = max(w, h) * 0.06
        ax.set_xlim(x1 - pad, x2 + pad)
        # KiCad-like screen view: Y grows downward.
        ax.set_ylim(y2 + pad, y1 - pad)
    ax.grid(True, linewidth=0.2, alpha=0.25)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel("X mm", fontsize=7)
    ax.set_ylabel("Y mm" if show_ylabel else "", fontsize=7)


def draw_board_bbox(ax: Any, bbox: Optional[BBox]) -> None:
    if not bbox:
        return
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            facecolor="none",
            edgecolor="0.35",
            linewidth=0.75,
            linestyle="--",
            zorder=1,
        )
    )


def draw_placement(
    ax: Any,
    task: Dict[str, Any],
    placement: Placement,
    board_bbox: Optional[BBox],
    view_bbox: Optional[BBox],
    title: str,
    labels: str,
    show_ylabel: bool,
    legend: bool = True,
) -> int:
    comp_by_ref, ref_to_module, colors = build_ref_maps(task)
    setup_axis(ax, title, view_bbox, show_ylabel=show_ylabel)
    draw_board_bbox(ax, board_bbox)

    polys = placement_polygons(task, placement)
    ncomp = len(comp_by_ref)
    drawn = 0

    for ref, poly, pl, size in polys:
        module = ref_to_module.get(ref, "unassigned")
        edge_color = colors.get(module, (0.05, 0.05, 0.05, 1.0))

        # Key requirement: no fill, border color only.
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                fill=False,
                facecolor="none",
                edgecolor=edge_color,
                linewidth=0.65,
                alpha=0.98,
                zorder=4,
            )
        )
        drawn += 1

        if should_label(ref, size, ncomp, labels):
            x, y, _ = pl
            ax.text(x, y, ref, fontsize=4.5, ha="center", va="center", color="0.05", zorder=5)

    if legend and colors:
        modules_sorted = sorted(colors.keys())
        # Keep legend compact; full mapping is still encoded in line colors.
        sample = modules_sorted[:10]
        handles = [plt.Line2D([0], [0], color=colors[m], lw=1.8, label=str(m)[:20]) for m in sample]
        ax.legend(handles=handles, loc="lower right", fontsize=5, framealpha=0.75, title="Modules", title_fontsize=6)

    return drawn


def make_row(
    board: str,
    manual_json: Path,
    infer_json: Path,
    task_json: Optional[Path],
    task: Dict[str, Any],
    infer: Dict[str, Any],
    manual_pl: Placement,
    infer_pl: Placement,
    use_raw: bool,
) -> Dict[str, Any]:
    infer_hpwl = metric_float(infer, "hpwl")
    manual_hpwl = metric_float(infer, "reference_hpwl")
    ratio = metric_float(infer, "hpwl_ratio")

    manual_calc = compute_hpwl_from_task(task, manual_pl) if manual_pl else None
    infer_calc = compute_hpwl_from_task(task, infer_pl) if infer_pl else None

    if manual_hpwl is None:
        manual_hpwl = manual_calc
    if infer_hpwl is None:
        infer_hpwl = infer_calc
    if ratio is None and manual_hpwl not in (None, 0) and infer_hpwl is not None:
        ratio = infer_hpwl / manual_hpwl

    comp_count = len(task.get("components", []) or [])
    _, ref_to_module, _ = build_ref_maps(task)
    module_count = len(set(ref_to_module.values()))

    return {
        "board": board,
        "component_count": comp_count,
        "module_count": module_count,
        "manual_refs": len(manual_pl),
        "infer_refs": len(infer_pl),
        "manual_hpwl": manual_hpwl,
        "infer_hpwl": infer_hpwl,
        "hpwl_ratio": ratio,
        "manual_hpwl_calc": manual_calc,
        "infer_hpwl_calc": infer_calc,
        "min_gap_p10": metric_float(infer, "min_gap_p10"),
        "min_gap_p25": metric_float(infer, "min_gap_p25"),
        "module_overlap_ratio": metric_float(infer, "module_overlap_ratio"),
        "local_density_p90": metric_float(infer, "local_density_p90"),
        "infer_placement_key": "placed_raw" if use_raw else "placed",
        "manual_json": str(manual_json),
        "task_json": str(task_json) if task_json else "",
        "infer_json": str(infer_json),
    }


def draw_board_page(
    pdf: Optional[PdfPages],
    out_png: Path,
    row: Dict[str, Any],
    task: Dict[str, Any],
    manual_pl: Placement,
    infer_pl: Placement,
    labels: str,
    dpi: int,
) -> None:
    manual_polys = placement_polygons(task, manual_pl)
    infer_polys = placement_polygons(task, infer_pl)
    board_bbox, view_bbox = derive_view_bbox(task, manual_polys, infer_polys)

    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[0.19, 0.81],
        width_ratios=[1, 1],
        hspace=0.08,
        wspace=0.08,
    )
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_m = fig.add_subplot(gs[1, 0])
    ax_i = fig.add_subplot(gs[1, 1])

    manual_hpwl = row.get("manual_hpwl")
    infer_hpwl = row.get("infer_hpwl")
    ratio = row.get("hpwl_ratio")

    pct = None
    try:
        if manual_hpwl and float(manual_hpwl) > 0 and infer_hpwl is not None:
            pct = (float(infer_hpwl) / float(manual_hpwl) - 1.0) * 100.0
    except Exception:
        pct = None

    hpwl_line = (
        f"Manual HPWL: {fmt_num(manual_hpwl, 2)}   |   "
        f"Infer HPWL: {fmt_num(infer_hpwl, 2)}   |   "
        f"Ratio: {fmt_num(ratio, 3)}"
    )
    if pct is not None:
        hpwl_line += f"   |   Infer vs manual: {pct:+.1f}%"

    ax_head.text(0.01, 0.92, row["board"], fontsize=13, weight="bold", va="top", ha="left")
    ax_head.text(0.01, 0.64, hpwl_line, fontsize=10.5, weight="bold", va="top", ha="left")
    ax_head.text(
        0.01,
        0.38,
        "Manual: JSON from infer_manual.  Infer: JSON from infer_out.  Rectangles are line-only; border color = module_id.",
        fontsize=7.5,
        va="top",
        ha="left",
        color="0.25",
    )
    ax_head.text(
        0.01,
        0.16,
        f"components={row['component_count']}   modules={row['module_count']}   "
        f"manual_refs={row['manual_refs']}   infer_refs={row['infer_refs']}   "
        f"infer_key={row['infer_placement_key']}",
        fontsize=7.5,
        va="top",
        ha="left",
        color="0.25",
    )

    draw_placement(
        ax=ax_m,
        task=task,
        placement=manual_pl,
        board_bbox=board_bbox,
        view_bbox=view_bbox,
        title="Manual layout - from infer_manual JSON",
        labels=labels,
        show_ylabel=True,
        legend=True,
    )
    draw_placement(
        ax=ax_i,
        task=task,
        placement=infer_pl,
        board_bbox=board_bbox,
        view_bbox=view_bbox,
        title="Infer layout - from infer output JSON",
        labels=labels,
        show_ylabel=False,
        legend=True,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_cover_page(pdf: PdfPages, rows: List[Dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.03, 0.94, "Manual vs Infer Layout Comparison - Module Color Line Drawing", fontsize=18, weight="bold", ha="left", va="top")
    ax.text(
        0.03,
        0.88,
        "Drawing rules:\n"
        "- Manual panel is read from infer_manual JSON, not KiCad geometry.\n"
        "- Infer panel is read from infer output JSON.\n"
        "- Components are drawn as rectangles with no fill color.\n"
        "- The same module_id uses the same border color in both Manual and Infer panels.\n"
        "- Dashed gray rectangle is board.bbox_mm when available.",
        fontsize=10,
        ha="left",
        va="top",
    )

    ax.text(0.03, 0.64, f"Matched boards: {len(rows)}", fontsize=12, weight="bold")

    if rows:
        sorted_rows = sorted(
            rows,
            key=lambda r: (
                999999 if r.get("hpwl_ratio") is None else float(r.get("hpwl_ratio")),
                str(r.get("board")),
            ),
            reverse=True,
        )
        cell_text = []
        for r in sorted_rows[:24]:
            cell_text.append([
                r.get("board", ""),
                str(r.get("component_count", "")),
                str(r.get("module_count", "")),
                str(r.get("manual_refs", "")),
                str(r.get("infer_refs", "")),
                fmt_num(r.get("manual_hpwl"), 1),
                fmt_num(r.get("infer_hpwl"), 1),
                fmt_num(r.get("hpwl_ratio"), 3),
            ])
        table = ax.table(
            cellText=cell_text,
            colLabels=["Board", "N", "Modules", "Manual", "Infer", "Manual HPWL", "Infer HPWL", "Ratio"],
            loc="lower left",
            bbox=[0.03, 0.06, 0.94, 0.52],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.auto_set_column_width(col=list(range(8)))
    else:
        ax.text(0.03, 0.55, "No matched boards found.", fontsize=12, color="red")

    pdf.savefig(fig, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys = [
        "board",
        "component_count",
        "module_count",
        "manual_refs",
        "infer_refs",
        "manual_hpwl",
        "infer_hpwl",
        "hpwl_ratio",
        "manual_hpwl_calc",
        "infer_hpwl_calc",
        "min_gap_p10",
        "min_gap_p25",
        "module_overlap_ratio",
        "local_density_p90",
        "infer_placement_key",
        "manual_json",
        "task_json",
        "infer_json",
        "png",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw clean manual-vs-infer PCB layout comparison from JSON.")
    ap.add_argument("--infer_dir", required=True, help="Directory containing model *.infer.json files.")
    ap.add_argument("--manual_root", default="infer_manual", help="Directory/root containing manual JSONs, default: infer_manual.")
    ap.add_argument("--task_root", default="data/infer", help="Directory/root containing base task JSONs for size/module/pads, default: data/infer.")
    ap.add_argument("--out_dir", default="plots/infer_layouts", help="Output directory for PNGs and summary CSV.")
    ap.add_argument("--pdf", default=None, help="Optional multipage PDF output.")
    ap.add_argument("--pattern", default="*.infer.json", help="Infer JSON glob under --infer_dir.")
    ap.add_argument("--raw", action="store_true", help="Use placed_raw instead of placed for infer panel.")
    ap.add_argument("--manual_key", default="auto", choices=["auto", "placed", "placed_raw", "expert", "manual", "component", "self"], help="Where to read manual placement from manual JSON.")
    ap.add_argument("--labels", choices=["none", "auto", "all"], default="auto", help="Reference label mode.")
    ap.add_argument("--dpi", type=int, default=180, help="PNG/PDF figure DPI.")
    ap.add_argument("--limit", type=int, default=0, help="Plot first N boards only. 0 = all.")
    args = ap.parse_args()

    infer_dir = Path(args.infer_dir)
    manual_root = Path(args.manual_root)
    task_root = Path(args.task_root) if args.task_root else None
    out_dir = Path(args.out_dir)

    if not infer_dir.exists():
        raise SystemExit(f"[error] infer_dir does not exist: {infer_dir}")
    if not manual_root.exists():
        raise SystemExit(f"[error] manual_root does not exist: {manual_root}")
    if task_root is not None and not task_root.exists():
        print(f"[warn] task_root does not exist, will use manual/infer JSON components if available: {task_root}")
        task_root = None

    out_dir.mkdir(parents=True, exist_ok=True)

    manual_index = build_json_index(manual_root)
    task_index = build_json_index(task_root)

    infer_files = sorted(infer_dir.rglob(args.pattern))
    if args.limit and args.limit > 0:
        infer_files = infer_files[:args.limit]
    if not infer_files:
        raise SystemExit(f"[error] no infer JSON under {infer_dir} with pattern {args.pattern}")

    rows: List[Dict[str, Any]] = []
    prepared = []
    skipped: List[str] = []

    for infer_path in infer_files:
        board = logical_stem(infer_path)
        manual_path = find_by_stem(board, manual_index)
        task_path = find_by_stem(board, task_index) if task_index else None

        if manual_path is None:
            msg = f"{infer_path} : no matching manual JSON under {manual_root}"
            skipped.append(msg)
            print(f"[skip] no matching manual JSON for {infer_path}")
            continue

        try:
            infer_doc = load_json(infer_path)
            manual_doc = load_json(manual_path)
            task_doc = load_json(task_path) if task_path else None
            task = choose_task_doc(manual_doc, infer_doc, task_doc)

            manual_pl = parse_json_placement(manual_doc, args.manual_key)
            infer_pl = parse_json_placement(infer_doc, "placed_raw" if args.raw else "placed")

            if not manual_pl:
                raise ValueError("manual JSON has no usable placement fields; expected placed/placed_raw/manual_placed or components[*].expert/manual.xy_mm")
            if not infer_pl:
                raise ValueError("infer JSON has no usable placed/placed_raw fields")
            if not isinstance(task.get("components"), list) or not task.get("components"):
                raise ValueError("no component list available for size/module drawing; pass --task_root data/infer or use manual JSON with components")

            row = make_row(
                board=board,
                manual_json=manual_path,
                infer_json=infer_path,
                task_json=task_path,
                task=task,
                infer=infer_doc,
                manual_pl=manual_pl,
                infer_pl=infer_pl,
                use_raw=args.raw,
            )
            row["png"] = str(out_dir / f"{board}.png")
            rows.append(row)
            prepared.append((row, task, manual_pl, infer_pl))
            print(f"[ok] {board}  manual={manual_path}  task={task_path or 'manual/infer JSON'}  infer={infer_path}")
        except Exception as e:
            msg = f"{infer_path} : {type(e).__name__}: {e}"
            skipped.append(msg)
            print(f"[skip] failed {infer_path}: {type(e).__name__}: {e}")

    prepared.sort(key=lambda x: str(x[0]["board"]))
    rows.sort(key=lambda r: str(r["board"]))

    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            draw_cover_page(pdf, rows)
            for row, task, manual_pl, infer_pl in prepared:
                draw_board_page(
                    pdf=pdf,
                    out_png=Path(row["png"]),
                    row=row,
                    task=task,
                    manual_pl=manual_pl,
                    infer_pl=infer_pl,
                    labels=args.labels,
                    dpi=args.dpi,
                )
    else:
        for row, task, manual_pl, infer_pl in prepared:
            draw_board_page(
                pdf=None,
                out_png=Path(row["png"]),
                row=row,
                task=task,
                manual_pl=manual_pl,
                infer_pl=infer_pl,
                labels=args.labels,
                dpi=args.dpi,
            )

    csv_path = out_dir / "layout_plot_summary.csv"
    write_summary_csv(rows, csv_path)

    skipped_path = None
    if skipped:
        skipped_path = out_dir / "plot_skipped.txt"
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")

    print("")
    print(f"[done] plotted={len(rows)} skipped={len(skipped)}")
    print(f"[out_dir] {out_dir}")
    print(f"[summary_csv] {csv_path}")
    if pdf_path:
        print(f"[pdf] {pdf_path}")
    if skipped_path:
        print(f"[skipped_list] {skipped_path}")


if __name__ == "__main__":
    main()
