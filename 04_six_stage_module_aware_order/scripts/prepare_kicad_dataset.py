#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcbplace.training_structure import generate_structure_file
from scripts.step1_kicad_to_json import build_layout_from_kicad, build_task_from_kicad


def _attach_expert_layout(task: dict, layout: dict) -> dict:
    placed = layout.get("placed") or {}
    missing = []
    for component in task.get("components") or []:
        ref = component.get("ref")
        pose = placed.get(ref)
        if pose is None:
            missing.append(ref)
            continue
        component["expert"] = {
            "xy_mm": [float(pose[0]), float(pose[1])],
            "rot": float(pose[2]),
        }
    if missing:
        raise ValueError(f"Missing expert placements for refs: {missing}")
    return task


def _drop_duplicate_component_refs(task: dict) -> list[str]:
    refs = [str(c.get("ref") or "") for c in task.get("components") or []]
    duplicates = sorted(ref for ref, count in Counter(refs).items() if count > 1)
    if not duplicates:
        return []

    duplicate_set = set(duplicates)
    task["components"] = [
        component
        for component in task.get("components") or []
        if str(component.get("ref") or "") not in duplicate_set
    ]
    for net, endpoints in list((task.get("nets") or {}).items()):
        kept = [
            endpoint
            for endpoint in endpoints
            if str(endpoint).split(".", 1)[0] not in duplicate_set
        ]
        if kept:
            task["nets"][net] = kept
        else:
            task["nets"].pop(net, None)
    return duplicates


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_split(
    source_dir: Path,
    output_dir: Path,
    audit_dir: Path,
    mode: str,
    grid_mm: float,
    interface_pad_mode: str,
) -> tuple[int, list[dict]]:
    boards = sorted(source_dir.glob("*.kicad_pcb"))
    if not boards:
        raise FileNotFoundError(f"No .kicad_pcb files in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, board in enumerate(boards, 1):
        task = build_task_from_kicad(
            board,
            grid_mm=grid_mm,
            interface_pad_mode=interface_pad_mode,
        )
        dropped_duplicate_refs = _drop_duplicate_component_refs(task)
        if not task.get("components"):
            raise ValueError(
                f"No placeable components parsed from {board}; "
                "check the KiCad syntax version and parser compatibility."
            )
        if mode == "train":
            task = _attach_expert_layout(
                task,
                build_layout_from_kicad(board, interface_pad_mode=interface_pad_mode),
            )

        intermediate = audit_dir / f"{board.stem}.source.json"
        output = output_dir / f"{board.stem}.json"
        audit = audit_dir / f"{board.stem}.audit.json"
        _write_json(intermediate, task)
        structured, report = generate_structure_file(
            intermediate,
            output,
            mode=mode,
            audit_out=audit,
        )
        summaries.append({
            "source": board.name,
            "output": output.name,
            "components": len(structured.get("components") or []),
            "modules": report.get("module_count"),
            "needs_review": report.get("semantic_needs_review_count"),
            "expert_region_input_leak": (report.get("expert_region_input_leak") or {}).get("leaked"),
            "dropped_duplicate_refs": dropped_duplicate_refs,
        })
        print(f"[{index}/{len(boards)}] {mode}: {board.name} -> {output.name}")
    return len(boards), summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert flat train/infer KiCad datasets into model-ready JSON.")
    parser.add_argument("--kicad-dir", default="data/kicad")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--grid-mm", type=float, default=1.0)
    parser.add_argument("--interface-pad-mode", choices=["numeric", "exclude", "all"], default="numeric")
    args = parser.parse_args()

    kicad_dir = Path(args.kicad_dir)
    data_dir = Path(args.data_dir)
    all_summaries = {}
    for mode in ("train", "infer"):
        count, summaries = convert_split(
            kicad_dir / mode,
            data_dir / mode,
            data_dir / "audits" / mode,
            mode,
            args.grid_mm,
            args.interface_pad_mode,
        )
        all_summaries[mode] = {"count": count, "boards": summaries}

    summary_path = data_dir / "conversion_summary.json"
    _write_json(summary_path, all_summaries)
    print(f"Wrote conversion summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
