import sys
from pathlib import Path
import argparse
import re
import math
from typing import Iterable, List, Optional, Tuple, Dict

# ensure project root in sys.path so "import pcbplace" works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcbplace.kicad_parser import parse_kicad_pcb
from pcbplace.utils import save_json, infer_type, is_pad_bbox_only_type, is_mounting_hole_type

RE_EXPERT_DIR = re.compile(r"^expert(\d+)$", re.IGNORECASE)
RE_BOARD_NAME = re.compile(r"processed_expertS(\d+)\.kicad_pcb$", re.IGNORECASE)

# ----------------------------------------------------------------------
# KiCad -> internal convention helpers
# ----------------------------------------------------------------------

def kicad_rot_to_ccw(rot_deg: float) -> float:
    """KiCad stores angles in a clockwise-positive convention in .kicad_pcb coordinates.
    Convert to standard math CCW degrees used by our downstream geometry (x right, y up).
    """
    try:
        return -float(rot_deg)
    except Exception:
        return 0.0


def _bbox_union(b1: Optional[Tuple[float, float, float, float]],
                b2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    if b1 is None:
        return b2
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))


def _bbox_from_points(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


INTERFACE_PAD_EXCLUDE_REGEX = r"^(MP|SH|SHIELD|CASE|GND|NC|HOLE|H\d+|MH\d+)"

def filter_interface_pads(pads: Iterable, mode: str = "numeric",
                          exclude_regex: str = INTERFACE_PAD_EXCLUDE_REGEX,
                          min_keep: int = 2):
    """Interface pad filtering to avoid mechanical/shield pads inflating bbox.

    mode:
      - all     : keep all pads
      - numeric : keep only pads whose name is purely numeric (1,2,3...)
      - exclude : drop pads matching exclude_regex (case-insensitive)

    If too few pads remain (< min_keep), fall back to keeping all pads.
    """
    pads = list(pads)
    if not pads:
        return []

    if mode == "all":
        return pads

    kept = []
    rx = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    for p in pads:
        name = str(getattr(p, "name", "")).strip()
        if mode == "numeric":
            if name.isdigit():
                kept.append(p)
        elif mode == "exclude":
            if rx is None or (not rx.search(name)):
                kept.append(p)
        else:
            kept.append(p)

    if len(kept) < int(min_keep):
        return pads
    return kept


def pad_bbox_local(pads: Iterable) -> Optional[Tuple[float, float, float, float]]:
    """Compute local bbox (in footprint coordinates) from a list of pads.

    Handles pad local rotation. NOTE: KiCad pad rot uses same convention as footprint,
    so we convert it with kicad_rot_to_ccw() before applying standard CCW rotation.
    """
    bbox: Optional[Tuple[float, float, float, float]] = None
    for p in pads:
        try:
            sx, sy = float(p.size[0]), float(p.size[1])
        except Exception:
            continue
        if sx <= 0 or sy <= 0:
            continue

        try:
            px, py = float(p.at[0]), float(p.at[1])
        except Exception:
            continue

        ang = math.radians(kicad_rot_to_ccw(getattr(p, "rot", 0.0)) % 360.0)
        ca, sa = math.cos(ang), math.sin(ang)
        corners = [(-sx/2, -sy/2), (-sx/2, sy/2), (sx/2, -sy/2), (sx/2, sy/2)]
        pts = []
        for x0, y0 in corners:
            xr = ca * x0 - sa * y0 + px
            yr = sa * x0 + ca * y0 + py
            pts.append((xr, yr))
        b = _bbox_from_points(pts)
        if b is not None:
            bbox = _bbox_union(bbox, b)
    return bbox


def choose_local_bbox(fp, typ: str, interface_pad_mode: str = "numeric") -> Optional[Tuple[float, float, float, float]]:
    """Choose which local bbox to use for size/center.

    - interface: pad bbox only (optionally filtered)
    - others: courtyard bbox if present, else pad bbox
    """
    if is_pad_bbox_only_type(typ):
        b = pad_bbox_local(filter_interface_pads(fp.pads, mode=interface_pad_mode))
        if b is not None:
            return b

    # Prefer explicit courtyard bbox if parser provides it
    b_courtyard = getattr(fp, "bbox_courtyard", None)
    b_pads = getattr(fp, "bbox_pads", None)

    if b_courtyard is not None:
        return b_courtyard
    if b_pads is not None:
        return b_pads

    # Backward compatibility: older parser only has bbox_local
    b = getattr(fp, "bbox_local", None)
    if b is not None:
        return b

    # Last resort: recompute from pads
    return pad_bbox_local(fp.pads)


# ----------------------------------------------------------------------
# Build task/layout json
# ----------------------------------------------------------------------

def build_task_from_kicad(kicad_pcb: Path, grid_mm: float, default_size_mm=(2.0, 2.0),
                          interface_pad_mode: str = "numeric") -> dict:
    text = kicad_pcb.read_text(encoding="utf-8", errors="ignore")
    board = parse_kicad_pcb(text)

    comps = []
    nets: Dict[str, List[str]] = {}

    for fp in board.footprints:
        typ = infer_type(fp.ref, fp.footprint)

        # Mounting holes are not model-placeable components.  Skip them before
        # adding components, pads, or net endpoints to the task JSON.
        if is_mounting_hole_type(typ):
            continue

        bbox = choose_local_bbox(fp, typ, interface_pad_mode=interface_pad_mode)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            w = float(max(0.1, x1 - x0))
            h = float(max(0.1, y1 - y0))
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
        else:
            w, h = default_size_mm
            cx, cy = 0.0, 0.0

        pads = []
        for p in fp.pads:
            pad_name = str(getattr(p, "name", ""))
            net_name = str(getattr(p, "net", ""))

            pads.append({
                "net": net_name,
                "rel_mm": [float(p.at[0] - cx), float(p.at[1] - cy)],
                # keep pad identity for later sequence/train steps
                "name": pad_name,
            })

            if net_name:
                nets.setdefault(net_name, []).append(f"{fp.ref}.{pad_name}" if pad_name else fp.ref)

        comps.append({
            "ref": fp.ref,
            "footprint": fp.footprint,
            "type": typ,
            "size_mm": [float(w), float(h)],
            "pads": pads,
            "allowed_sides": [],
            "fixed": False,
        })

    # (optional) filter nets with <2 pins if you want a cleaner graph later
    # nets = {k: v for k, v in nets.items() if len(v) >= 2}

    task = {
        "board": {"bbox_mm": list(board.bbox), "grid_mm": float(grid_mm)},
        "components": comps,
        "nets": nets,
        "graph": {},
        "meta": {
            "kicad_rotation_converted_to_ccw": True,
            "interface_bbox_source": "pads",
            "interface_pad_filter_mode": interface_pad_mode,
        }
    }
    return task


def build_layout_from_kicad(kicad_pcb: Path, interface_pad_mode: str = "numeric") -> dict:
    text = kicad_pcb.read_text(encoding="utf-8", errors="ignore")
    board = parse_kicad_pcb(text)

    placed = {}
    for fp in board.footprints:
        if not fp.ref:
            continue

        typ = infer_type(fp.ref, fp.footprint)

        # Keep expert/layout refs aligned with task JSON: mounting holes are
        # ignored by train/infer and must not appear as expert targets.
        if is_mounting_hole_type(typ):
            continue

        # convert KiCad rotation to our CCW convention
        r = kicad_rot_to_ccw(float(fp.rot))

        x = float(fp.at[0])
        y = float(fp.at[1])

        # IMPORTANT: export placement as component CENTER
        bbox = choose_local_bbox(fp, typ, interface_pad_mode=interface_pad_mode)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            th = math.radians(r % 360.0)
            c = math.cos(th)
            s = math.sin(th)

            # rotate local center offset into global
            x += cx * c - cy * s
            y += cx * s + cy * c

        placed[fp.ref] = [x, y, float(r)]

    return {"placed": placed}


def pick_board_file(expert_dir: Path) -> Path | None:
    # find all boards under expert_dir (recursive)
    boards = sorted(expert_dir.rglob("*.kicad_pcb"))
    if not boards:
        return None

    # prefer processed_expertS{N}.kicad_pcb where N comes from expert{N}
    m = RE_EXPERT_DIR.match(expert_dir.name)
    if m:
        n = m.group(1)
        prefer = f"processed_expertS{n}.kicad_pcb".lower()
        for b in boards:
            if b.name.lower() == prefer:
                return b

    # else prefer any processed_expertS*.kicad_pcb
    for b in boards:
        if RE_BOARD_NAME.search(b.name):
            return b

    # fallback: first one
    return boards[0]


def process_expert(expert_dir: Path, out_dir: Path, grid_mm: float, interface_pad_mode: str):
    board = pick_board_file(expert_dir)
    if board is None:
        print(f"[SKIP] no .kicad_pcb under {expert_dir}")
        return

    expert_name = expert_dir.name

    # put outputs into a subfolder per expert
    out_subdir = out_dir / expert_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    out_task = out_subdir / f"{expert_name}_task.json"
    out_layout = out_subdir / f"{expert_name}_layout.json"

    task = build_task_from_kicad(board, grid_mm=grid_mm, interface_pad_mode=interface_pad_mode)
    layout = build_layout_from_kicad(board, interface_pad_mode=interface_pad_mode)

    save_json(task, str(out_task))
    save_json(layout, str(out_layout))

    print(f"[OK] {expert_name}: {board} -> {out_subdir}\\{out_task.name}, {out_subdir}\\{out_layout.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kicad_root", required=True, help=r'kicad根目录(含expert*) 或 单个expert目录')
    ap.add_argument("--out_dir", required=True, help=r'输出目录，例如 examples\\json')
    ap.add_argument("--grid_mm", type=float, default=1.0)
    ap.add_argument("--only_expert", default="", help="可选：只处理某个expert，例如 expert1")
    ap.add_argument("--interface_pad_mode", default="numeric", choices=["numeric","exclude","all"],
                    help="接口器件 bbox 仅用 pads 时，pads 过滤策略（默认 numeric）")
    args = ap.parse_args()

    kicad_root = Path(args.kicad_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # if user points to an expert dir directly
    if RE_EXPERT_DIR.match(kicad_root.name) and kicad_root.is_dir():
        if args.only_expert and kicad_root.name.lower() != args.only_expert.lower():
            print(f"[SKIP] only_expert={args.only_expert}, got {kicad_root.name}")
            return
        process_expert(kicad_root, out_dir, args.grid_mm, args.interface_pad_mode)
        return

    # else treat as root containing expert dirs
    experts = sorted([p for p in kicad_root.iterdir() if p.is_dir() and RE_EXPERT_DIR.match(p.name)])
    if not experts:
        raise SystemExit(f"No expert* folders under: {kicad_root}")

    for e in experts:
        if args.only_expert and e.name.lower() != args.only_expert.lower():
            continue
        process_expert(e, out_dir, args.grid_mm, args.interface_pad_mode)

if __name__ == "__main__":
    main()

