from __future__ import annotations

import glob
import json
from pathlib import Path


def get_scale(data: dict) -> float:
    bbox = data["board"]["bbox_mm"]
    width = abs(float(bbox[2]) - float(bbox[0]))
    height = abs(float(bbox[3]) - float(bbox[1]))
    max_dim = max(width, height)

    # Some source files store coordinates in nanometre-like units.
    if max_dim > 1_000_000:
        return 1e-6

    # SuperTrouble2017_rev uses approximately 0.1 mm units.
    if max_dim > 1_000:
        return 0.1

    return 1.0


def scale_pair(value, scale):
    if not isinstance(value, list) or len(value) != 2:
        return value
    return [
        float(value[0]) * scale,
        float(value[1]) * scale,
    ]


def fix_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scale = get_scale(data)
    if scale == 1.0:
        return False

    bbox = data["board"]["bbox_mm"]
    old_width = float(bbox[2]) - float(bbox[0])
    old_height = float(bbox[3]) - float(bbox[1])

    data["board"]["bbox_mm"] = [
        float(v) * scale for v in bbox
    ]

    for comp in data.get("components", []):
        if "size_mm" in comp:
            comp["size_mm"] = scale_pair(
                comp["size_mm"],
                scale,
            )

        for pad in comp.get("pads", []):
            if "rel_mm" in pad:
                pad["rel_mm"] = scale_pair(
                    pad["rel_mm"],
                    scale,
                )

    meta = data.setdefault("meta", {})
    meta["unit_scale_applied"] = scale
    meta["original_board_size"] = [
        old_width,
        old_height,
    ]

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    new_bbox = data["board"]["bbox_mm"]
    new_width = new_bbox[2] - new_bbox[0]
    new_height = new_bbox[3] - new_bbox[1]

    print(
        f"fixed {path.name}: "
        f"scale={scale:g}, "
        f"{old_width:g}x{old_height:g} -> "
        f"{new_width:.3f}x{new_height:.3f} mm"
    )

    return True


def main():
    paths = sorted(
        Path(p)
        for p in glob.glob(
            "data/benchmark_infer/*.json"
        )
    )

    fixed = 0
    for path in paths:
        if fix_file(path):
            fixed += 1

    print(f"checked={len(paths)}, fixed={fixed}")


if __name__ == "__main__":
    main()
