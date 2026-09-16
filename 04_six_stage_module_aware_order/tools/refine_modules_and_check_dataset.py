#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild every data/{train,infer} board with v2 fine modules and audit it.

The script deliberately regenerates from each audit ``*.source.json`` when it
exists, so stale v1 module annotations/prior regions cannot survive a rebuild.
It writes fresh board JSON, companion structure audit JSON, and a compact
legality/quality report.  Train boards receive a full CPU expert-guided legal
rollout; infer boards receive schema, partition and first-prefix mask checks.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.json_schema import validate_task_json
from pcbplace.training_structure import generate_structure_file


INTERFACE_TYPE_TOKENS = (
    "conn_", "connector", "header", "terminal", "usb", "rj", "jack",
    "fpc", "ffc", "socket", "sma", "coax",
)
INTERFACE_PREFIXES = {"J", "P", "JP", "CN", "CON", "CONN", "RJ", "X", "XA"}


def _prefix(ref: str) -> str:
    import re
    m = re.match(r"[A-Za-z]+", str(ref or ""))
    return m.group(0).upper() if m else ""


def _physical_interface(c: Dict[str, Any]) -> bool:
    typ = str(c.get("type") or "").lower()
    sem = str(c.get("semantic_class") or (c.get("semantic") or {}).get("semantic_class") or "").lower()
    role = str(c.get("placement_role") or (c.get("semantic") or {}).get("placement_role") or "").lower()
    ref = str(c.get("ref") or "")
    if typ.startswith("conn_") or any(tok in typ for tok in INTERFACE_TYPE_TOKENS):
        return True
    return sem in {"interface", "mechanical_edge_interface", "rf"} and (_prefix(ref) in INTERFACE_PREFIXES or "edge" in role)


def _signal_nets(c: Dict[str, Any]) -> set[str]:
    power_tokens = ("GND", "GROUND", "VCC", "VDD", "VBUS", "VIN", "VOUT", "VBAT", "PWR", "POWER", "3V", "5V", "12V", "24V")
    out = set()
    for pad in c.get("pads") or []:
        net = str((pad or {}).get("net") or "").strip()
        up = net.upper()
        if net and not any(t in up for t in power_tokens):
            out.add(net)
    return out


def _bbox_inside(bb: Any, board: Iterable[float], eps: float = 1e-6) -> bool:
    if not isinstance(bb, list) or len(bb) != 4:
        return False
    try:
        x0, y0, x1, y1 = map(float, bb)
        bx0, by0, bx1, by1 = map(float, board)
    except Exception:
        return False
    return x1 > x0 and y1 > y0 and x0 >= bx0 - eps and y0 >= by0 - eps and x1 <= bx1 + eps and y1 <= by1 + eps


def quality_checks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    board = (data.get("board") or {}).get("bbox_mm") or []
    comp_by_ref = {str(c.get("ref")): c for c in data.get("components") or [] if isinstance(c, dict) and c.get("ref")}
    seen: Dict[str, str] = {}
    for mod in data.get("modules") or []:
        if not isinstance(mod, dict):
            issues.append({"kind": "module_not_object"})
            continue
        mid = str(mod.get("module_id") or "")
        members = [str(x) for x in mod.get("members") or []]
        for ref in members:
            if ref not in comp_by_ref:
                issues.append({"kind": "module_unknown_member", "module": mid, "ref": ref})
            elif ref in seen:
                issues.append({"kind": "module_duplicate_member", "ref": ref, "modules": [seen[ref], mid]})
            else:
                seen[ref] = mid
        prior = mod.get("prior_region") if isinstance(mod.get("prior_region"), dict) else {}
        bb = prior.get("bbox_mm")
        if not _bbox_inside(bb, board):
            issues.append({"kind": "prior_bbox_invalid_or_outside_board", "module": mid, "bbox_mm": bb})
        src = str(prior.get("source") or "")
        if not src.startswith("rule_v2_"):
            issues.append({"kind": "stale_or_unknown_prior_source", "module": mid, "source": src})
        if str(mod.get("module_type") or "") == "interface":
            anchor = comp_by_ref.get(str(mod.get("anchor_ref") or ""), {})
            if not _physical_interface(anchor):
                issues.append({"kind": "interface_module_without_physical_connector_anchor", "module": mid, "anchor_ref": mod.get("anchor_ref")})
            anchor_nets = _signal_nets(anchor)
            for ref in members:
                if ref == mod.get("anchor_ref"):
                    continue
                c = comp_by_ref.get(ref, {})
                sem = str(c.get("semantic_class") or "").lower()
                typ = str(c.get("type") or "").lower()
                ref_upper = str(c.get("ref") or "").upper()
                direct_support = bool(anchor_nets.intersection(_signal_nets(c)))
                protection = any(token in f"{ref_upper} {typ}" for token in ("TVS", "ESD", "PROTECT", "CHOKE", "FERRITE"))
                if (sem not in {"interface_support", "passive"} and not protection) or not direct_support:
                    issues.append({
                        "kind": "interface_module_foreign_member",
                        "module": mid,
                        "anchor_ref": mod.get("anchor_ref"),
                        "ref": ref,
                        "semantic_class": sem,
                    })
    missing = sorted(set(comp_by_ref) - set(seen))
    for ref in missing:
        issues.append({"kind": "component_without_module", "ref": ref})
    return issues


def _nearest_legal(mask: np.ndarray, x: float, y: float, rot: float, bbox: List[float], grid: float, rotations: Tuple[int, ...]) -> Tuple[int, int, int] | None:
    xmin, ymin = float(bbox[0]), float(bbox[1])
    best: Tuple[float, int, int, int] | None = None
    for ri in range(mask.shape[0]):
        pts = np.argwhere(mask[ri] > 0.5)
        if pts.size == 0:
            continue
        xs = xmin + (pts[:, 0].astype(float) + 0.5) * grid
        ys = ymin + (pts[:, 1].astype(float) + 0.5) * grid
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        rot_penalty = ((((float(rotations[ri]) - float(rot) + 180.0) % 360.0) - 180.0) / 90.0) ** 2
        k = int(np.argmin(d2))
        score = float(d2[k] + 0.25 * grid * grid * rot_penalty)
        candidate = (score, ri, int(pts[k, 0]), int(pts[k, 1]))
        if best is None or candidate < best:
            best = candidate
    return None if best is None else (best[1], best[2], best[3])


def train_expert_legality(path: Path) -> Dict[str, Any]:
    """Prove a complete legal route using the board's train-only expert labels."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_ref = {str(c["ref"]): c for c in data.get("components") or []}
    task = task_from_json(str(path), sequence_policy="rebuild", load_expert=False)
    env = PlacementEnv(task)
    bbox = list(data["board"]["bbox_mm"])
    grid = float(data["board"].get("grid_mm", 1.0))
    for t in range(len(env.sequence)):
        ref = env.current_ref()
        c = by_ref.get(str(ref), {})
        ex = c.get("expert") if isinstance(c.get("expert"), dict) else None
        if not ex or not isinstance(ex.get("xy_mm"), list) or len(ex["xy_mm"]) != 2:
            return {"ok": False, "reason": "missing_expert_label", "t": t, "ref": ref}
        mask, _ = env.action_mask_and_bias(str(ref))
        action = _nearest_legal(mask, float(ex["xy_mm"][0]), float(ex["xy_mm"][1]), float(ex.get("rot", 0.0)), bbox, grid, env.rotations)
        if action is None:
            return {"ok": False, "reason": "no_legal_action", "t": t, "ref": ref, "type": c.get("type"), "size_mm": c.get("size_mm")}
        _, _, done, info = env.step(action, assume_legal=True, return_observation=False, compute_objective=False)
        if info.get("illegal"):
            return {"ok": False, "reason": str(info.get("reason") or "postcheck"), "t": t, "ref": ref, "action": list(action)}
        if done and t + 1 < len(env.sequence):
            return {"ok": False, "reason": "terminated_early", "t": t, "ref": ref}
    return {"ok": True, "steps": len(env.sequence)}


def infer_prefix_legality(path: Path, max_steps: int = 5) -> Dict[str, Any]:
    """Verify that infer boards offer a legal action through their early prefix.

    Full infer feasibility cannot be certified without an expert route; this
    check therefore reports only true empty-mask failures, not model quality.
    """
    task = task_from_json(str(path), sequence_policy="rebuild", load_expert=False)
    env = PlacementEnv(task)
    steps = min(max(0, int(max_steps)), len(env.sequence))
    for t in range(steps):
        ref = env.current_ref()
        mask, bias = env.action_mask_and_bias(str(ref))
        if not np.any(mask > 0.5):
            c = env.comp_by_ref[str(ref)]
            return {"ok": False, "reason": "no_legal_action", "t": t, "ref": ref, "type": c.type, "size_mm": list(c.size_mm)}
        valid = np.argwhere(mask > 0.5)
        # Deterministic, prior-aware smoke action: maximum bias over legal cells.
        ri, ix, iy = max((tuple(v) for v in valid), key=lambda a: float(bias[a]))
        _, _, _done, info = env.step((int(ri), int(ix), int(iy)), assume_legal=True, return_observation=False, compute_objective=False)
        if info.get("illegal"):
            return {"ok": False, "reason": str(info.get("reason") or "postcheck"), "t": t, "ref": ref}
    return {"ok": True, "steps": steps}


def iter_board_paths(data_root: Path, splits: List[str]) -> Iterable[Tuple[str, Path, Path]]:
    for split in splits:
        board_dir = data_root / split
        for out_path in sorted(board_dir.glob("*.json")):
            source = data_root / "audits" / split / f"{out_path.stem}.source.json"
            yield split, out_path, source if source.exists() else out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--splits", default="train,infer")
    ap.add_argument("--report-json", default="data/module_refinement_v2_report.json")
    ap.add_argument("--report-md", default="data/module_refinement_v2_report.md")
    ap.add_argument("--infer-prefix-steps", type=int, default=5)
    ap.add_argument("--skip-legality", action="store_true", help="Rebuild and run static checks only; omit rollout checks.")
    args = ap.parse_args()

    data_root = (ROOT / args.data_root).resolve()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    boards: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for split, out_path, input_path in iter_board_paths(data_root, splits):
        audit_path = data_root / "audits" / split / f"{out_path.stem}.audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {"split": split, "file": str(out_path.relative_to(ROOT)), "source": str(input_path.relative_to(ROOT))}
        try:
            # Atomic destination avoids leaving a partial board if a later check fails.
            with tempfile.NamedTemporaryFile(prefix=f"{out_path.stem}.", suffix=".json", dir=str(out_path.parent), delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                structured, _audit = generate_structure_file(input_path, tmp_path, mode=split, audit_out=audit_path)
                validate_task_json(structured, source=str(out_path))
                tmp_path.replace(out_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            issues = quality_checks(data)
            row.update({
                "components": len(data.get("components") or []),
                "modules": len(data.get("modules") or []),
                "module_types": dict(Counter(str(m.get("module_type")) for m in data.get("modules") or [])),
                "quality_issues": issues,
            })
            if args.skip_legality:
                row["legality"] = {"ok": None, "reason": "not_run"}
            else:
                row["legality"] = train_expert_legality(out_path) if split == "train" else infer_prefix_legality(out_path, args.infer_prefix_steps)
        except Exception as exc:
            row.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5)})
            errors.append(row)
        boards.append(row)
        status = "ERROR" if row.get("error") else ("OK" if row.get("legality", {}).get("ok") and not row.get("quality_issues") else "REVIEW")
        print(f"[{status}] {row['file']}", flush=True)

    quality_issues = [
        {"file": b["file"], **issue}
        for b in boards for issue in b.get("quality_issues", [])
    ]
    legality_failures = [
        {"file": b["file"], **b.get("legality", {})}
        for b in boards if b.get("legality", {}).get("ok") is False
    ]
    report = {
        "generator": "refine_modules_and_check_dataset.py",
        "module_partition": "fine_modules_v2",
        "prior_generator": "rule_v2_*",
        "processed_boards": len(boards),
        "splits": splits,
        "generation_errors": errors,
        "quality_issue_count": len(quality_issues),
        "quality_issues": quality_issues,
        "legality_failure_count": len(legality_failures),
        "legality_failures": legality_failures,
        "boards": boards,
        "notes": {
            "train_legality": "full CPU expert-guided legal rollout with global nearest legal snap at every sequence step",
            "infer_legality": f"first {args.infer_prefix_steps} steps using deterministic prior-aware legal actions; not a complete packing proof",
            "prior_rule": "v2 emits a narrow edge prior only for a physical connector with an explicit edge-side constraint; otherwise interface/power hints are broad and low-confidence",
        },
    }
    report_json = (ROOT / args.report_json).resolve()
    report_md = (ROOT / args.report_md).resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Module Refinement v2 — Dataset Report",
        "",
        f"- Processed boards: **{len(boards)}**",
        f"- Generation errors: **{len(errors)}**",
        f"- Static partition/prior issues: **{len(quality_issues)}**",
        f"- Legality failures: **{len(legality_failures)}**",
        "",
        "## Checks",
        "",
        "- All board JSON files were rebuilt from audit source JSON where available.",
        "- Every component must belong to exactly one module.",
        "- Interface modules must have a physical connector anchor; non-anchor members must be directly connected passive/interface-support parts.",
        "- Every module prior must be a board-contained `rule_v2_*` prior.",
        "- Train boards use a complete CPU expert-guided legal rollout; infer boards use a five-step prior-aware mask smoke test.",
        "",
        "## Failures requiring review",
        "",
    ]
    if errors or quality_issues or legality_failures:
        for item in errors + quality_issues + legality_failures:
            lines.append(f"- `{item.get('file', '')}` — `{item.get('kind') or item.get('reason') or item.get('error')}`")
    else:
        lines.append("None.")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "processed_boards": len(boards),
        "generation_errors": len(errors),
        "quality_issues": len(quality_issues),
        "legality_failures": len(legality_failures),
        "report_json": str(report_json),
    }, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
