from __future__ import annotations

import json

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.policy_runtime import build_context_ref_order


def _write_task(path):
    task = {
        "board": {"bbox_mm": [0, 0, 30, 20], "grid_mm": 1.0},
        "components": [
            {
                "ref": "U1",
                "type": "chip",
                "size_mm": [4.0, 4.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
                "allowed_sides": [],
                "fixed": True,
                "fixed_xy_mm": [10.5, 10.5],
                "fixed_rot": 90,
            },
            {
                "ref": "C1",
                "type": "capacitor",
                "size_mm": [2.0, 2.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
                "allowed_sides": [],
            },
            {
                "ref": "R1",
                "type": "resistor",
                "size_mm": [2.0, 2.0],
                "pads": [{"net": "N2", "rel_mm": [0.0, 0.0]}],
                "allowed_sides": [],
            },
        ],
        "nets": {"N1": ["U1.1", "C1.1"], "N2": ["R1.1"]},
        "graph": {"sequence": ["U1", "C1", "R1"]},
    }
    path.write_text(json.dumps(task), encoding="utf-8")


def test_fixed_component_is_preplaced_skipped_and_blocks_mask(tmp_path):
    path = tmp_path / "task.json"
    _write_task(path)
    task = task_from_json(str(path), sequence_policy="stored", load_expert=False)
    env = PlacementEnv(task, min_spacing_mm=0.0)

    assert env.fixed_refs == ["U1"]
    assert env.sequence == ["C1", "R1"]
    assert env.placed["U1"] == (10.5, 10.5, 90)
    assert env.current_ref() == "C1"

    mask, _bias = env.action_mask_and_bias("C1")
    xmin, ymin, _xmax, _ymax = task.bbox_mm
    ix = int(round((10.5 - xmin) / task.grid_mm - 0.5))
    iy = int(round((10.5 - ymin) / task.grid_mm - 0.5))
    assert mask[:, ix, iy].max() == 0.0


def test_fixed_component_is_context_token_for_connected_dynamic_ref(tmp_path):
    path = tmp_path / "task.json"
    _write_task(path)
    task = task_from_json(str(path), sequence_policy="stored", load_expert=False)
    env = PlacementEnv(task, min_spacing_mm=0.0)

    ctx = build_context_ref_order(env, "C1", max_tokens=8)
    assert "U1" in ctx
