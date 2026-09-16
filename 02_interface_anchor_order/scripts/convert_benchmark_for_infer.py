from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from pcbplace.json_schema import validate_task_json
from pcbplace.sequence_heuristic import generate_sequence
from pcbplace.utils import infer_type


def convert_one(src: Path) -> dict:
    with src.open("r", encoding="utf-8") as f:
        old = json.load(f)

    board = old["board"]

    x0 = float(board.get("x_min", 0.0))
    y0 = float(board.get("y_min", 0.0))
    width = float(board.get("width", board.get("w")))
    height = float(board.get("height", board.get("h")))

    components = []
    comp_by_ref = {}

    for comp in old["components"]:
        ref = str(
            comp.get("ref")
            or comp.get("id")
            or comp.get("name")
        )

        footprint = str(comp.get("footprint", ""))

        new_comp = {
            "ref": ref,
            "footprint": footprint,
            "type": str(
                comp.get("type")
                or infer_type(ref, footprint)
            ),
            "size_mm": [
                float(comp.get("width", comp.get("w", 1.0))),
                float(comp.get("height", comp.get("h", 1.0))),
            ],
            "pads": [],
            "allowed_sides": list(
                comp.get("allowed_sides") or []
            ),
            "fixed": bool(comp.get("fixed", False)),
        }

        components.append(new_comp)
        comp_by_ref[ref] = new_comp

    nets = {}
    pad_count = {ref: 0 for ref in comp_by_ref}

    for net_obj in old.get("nets", []):
        net_name = str(net_obj.get("name", ""))

        if not net_name:
            continue

        net_pins = []

        for pin in net_obj.get("pins", []):
            ref = str(
                pin.get("ref")
                or pin.get("component")
                or pin.get("component_id")
                or ""
            )

            if ref not in comp_by_ref:
                continue

            pad_count[ref] += 1

            pad_name = str(
                pin.get("pad")
                or pin.get("pin")
                or pad_count[ref]
            )

            dx = float(
                pin.get(
                    "dx",
                    pin.get("offset_x", 0.0),
                )
            )
            dy = float(
                pin.get(
                    "dy",
                    pin.get("offset_y", 0.0),
                )
            )

            comp_by_ref[ref]["pads"].append({
                "net": net_name,
                "rel_mm": [dx, dy],
                "name": pad_name,
            })

            net_pins.append(f"{ref}.{pad_name}")

        if net_pins:
            nets[net_name] = net_pins

    task = {
        "board": {
            "bbox_mm": [
                x0,
                y0,
                x0 + width,
                y0 + height,
            ],
            "grid_mm": float(board.get("grid_mm", 1.0)),
        },
        "components": components,
        "nets": nets,
        "graph": {},
        "meta": {
            "converted_from": str(src),
            "source_format": old.get("source_format"),
        },
    }

    sequence, sequence_meta = generate_sequence(
        task,
        bfs_depth=3,
        return_meta=True,
    )

    task["graph"] = {
        "sequence": sequence,
        "sequence_source":
            "heuristic_anchor_interface_priority_v6",
        "sequence_config": sequence_meta,
    }

    validate_task_json(task, source=str(src))

    if len(sequence) != len(components):
        raise ValueError(
            f"{src}: sequence length {len(sequence)} "
            f"!= component count {len(components)}"
        )

    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_glob", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    paths = sorted(
        Path(path)
        for path in glob.glob(args.in_glob)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in paths:
        dst = out_dir / src.name
        task = convert_one(src)

        with dst.open("w", encoding="utf-8") as f:
            json.dump(
                task,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"{src} -> {dst} "
            f"components={len(task['components'])} "
            f"nets={len(task['nets'])}"
        )

    print(f"converted {len(paths)} files")


if __name__ == "__main__":
    main()
