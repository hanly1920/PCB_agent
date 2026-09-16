from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.env_cuda import _coerce_rwh_map, _coerce_wh_map, _compute_pitch_delta, objective_delta_mask_cuda


def test_coerce_rwh_map_accepts_transposed_anisotropic_wire_map():
    r, w, h = 4, 24, 1080
    transposed = torch.zeros(r, h, w)
    out = _coerce_rwh_map(transposed, r, w, h, "hpwl")
    assert tuple(out.shape) == (r, w, h)


def test_objective_delta_mask_cuda_uses_canonical_rwh_on_tall_board_cpu():
    payload = {
        "board": {"bbox_mm": [0.0, 0.0, 24.0, 108.0], "grid_mm": 1.0},
        "components": [
            {
                "ref": "U1",
                "type": "IC",
                "size_mm": [1.0, 1.0],
                "pads": [
                    {"net": "N1", "rel_mm": [0.0, 0.0]},
                    {"net": "N2", "rel_mm": [0.2, 0.0]},
                ],
            },
            {
                "ref": "C1",
                "type": "C",
                "size_mm": [1.0, 1.0],
                "pads": [
                    {"net": "N1", "rel_mm": [0.0, 0.0]},
                    {"net": "N2", "rel_mm": [-0.2, 0.0]},
                ],
            },
        ],
        "nets": {"N1": ["U1", "C1"], "N2": ["U1", "C1"]},
        "graph": {"sequence": ["U1", "C1"]},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "task.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        env = PlacementEnv(task_from_json(str(path), sequence_policy="stored"))
        env.placed["U1"] = (12.0, 10.0, 0)
        env.placed_order = ["U1"]
        env.t = 1

        maps = objective_delta_mask_cuda(env, "C1", torch.device("cpu"))
        assert tuple(maps["hpwl"].shape) == (4, 24, 108)
        assert tuple(maps["nslw"].shape) == (4, 24, 108)
        assert tuple(maps["region"].shape) == (4, 24, 108)
        assert tuple(maps["total"].shape) == (4, 24, 108)


def test_coerce_wh_map_accepts_axis_and_transposed_maps():
    w, h = 24, 1080
    assert tuple(_coerce_wh_map(torch.zeros(h, w), w, h, "transposed").shape) == (w, h)
    assert tuple(_coerce_wh_map(torch.zeros(w, 1), w, h, "x_axis").shape) == (w, h)
    assert tuple(_coerce_wh_map(torch.zeros(1, h), w, h, "y_axis").shape) == (w, h)



def test_compute_pitch_delta_handles_full_grid_on_extreme_board_cpu():
    components = []
    for i in range(4):
        components.append({
            "ref": f"P{i+1}",
            "type": "R",
            "size_mm": [0.4, 0.4],
            "pitch_group": "PG",
            "row_axis": "x",
            "pads": [],
        })
    payload = {
        "board": {"bbox_mm": [0.0, 0.0, 2.4, 108.0], "grid_mm": 0.1},
        "components": components,
        "nets": {},
        "graph": {"sequence": ["P1", "P2", "P3", "P4"]},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "task.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        env = PlacementEnv(task_from_json(str(path), sequence_policy="stored"))
        env.placed = {
            "P1": (0.4, 10.0, 0),
            "P2": (0.9, 10.0, 0),
            "P3": (1.4, 10.0, 0),
        }
        env.placed_order = ["P1", "P2", "P3"]
        env.t = 3

        w, h = env.grid_shape()
        ix = torch.arange(w, dtype=torch.float32).view(w, 1)
        iy = torch.arange(h, dtype=torch.float32).view(1, h)
        Xc = (ix + 0.5) * float(env.task.grid_mm)
        Yc = (iy + 0.5) * float(env.task.grid_mm)
        X_full = Xc.expand(w, h)
        Y_full = Yc.expand(w, h)

        out = _compute_pitch_delta(env, "P4", X_full, Y_full, torch.device("cpu"))
        assert tuple(out.shape) == (w, h)
