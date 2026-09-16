from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


PAIR_KEYS = {
    "at_mm",
    "pos_mm",
    "position_mm",
    "size_mm",
    "wh_mm",
    "center_mm",
    "xy_mm",
    "offset_mm",
}

SCALAR_KEYS = {
    "width_mm",
    "height_mm",
    "radius_mm",
    "clearance_mm",
    "margin_mm",
    "grid_mm",
}


def detect_scale(data: dict) -> float:
    bbox = data.get("board", {}).get("bbox_mm")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 1.0

    width = abs(float(bbox[2]) - float(bbox[0]))
    height = abs(float(bbox[3]) - float(bbox[1]))
    max_dim = max(width, height)

    # Nanometre-like coordinates being interpreted as millimetres.
    if max_dim > 1_000_000:
        return 1e-6

    # Some datasets use approximately 0.1 mm integer units.
    if max_dim > 1_000:
        return 0.1

    return 1.0


def scale_tree(value, scale: float, parent_key: str | None = None):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "bbox_mm" and isinstance(item, list) and len(item) == 4:
                result[key] = [float(v) * scale for v in item]
            elif key in PAIR_KEYS and isinstance(item, list) and len(item) == 2:
                result[key] = [float(v) * scale for v in item]
            elif key in SCALAR_KEYS and isinstance(item, (int, float)):
                result[key] = float(item) * scale
            elif key.endswith("_mm") and isinstance(item, (int, float)):
                result[key] = float(item) * scale
            else:
                result[key] = scale_tree(item, scale, key)
        return result

    if isinstance(value, list):
        return [scale_tree(item, scale, parent_key) for item in value]

    return value


def fix_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scale = detect_scale(data)
    if scale == 1.0:
        return False

    old_bbox = data["board"]["bbox_mm"]
    old_w = float(old_bbox[2]) - float(old_bbox[0])
    old_h = float(old_bbox[3]) - float(old_bbox[1])

    fixed = scale_tree(data, scale)

    new_bbox = fixed["board"]["bbox_mm"]
    new_w = float(new_bbox[2]) - float(new_bbox[0])
    new_h = float(new_bbox[3]) - float(new_bbox[1])

    with path.open("w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(
        f"fixed {path.name}: scale={scale:g}, "
        f"{old_w:g}x{old_h:g} -> {new_w:g}x{new_h:g} mm"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    fixed_count = 0

    for filename in paths:
        fixed_count += int(fix_file(Path(filename)))

    print(f"checked={len(paths)}, fixed={fixed_count}")


if __name__ == "__main__":
    main()
