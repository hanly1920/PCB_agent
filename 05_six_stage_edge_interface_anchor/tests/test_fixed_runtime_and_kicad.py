from __future__ import annotations

import json
import os

from pcbplace.infer import materialize_fixed_refs_task_json
from pcbplace.kicad_parser import parse_kicad_pcb


def test_materialize_fixed_refs_promotes_current_pose(tmp_path):
    task = {
        "board": {"bbox_mm": [0, 0, 20, 20], "grid_mm": 1.0},
        "components": [
            {
                "ref": "U1",
                "type": "chip",
                "size_mm": [4.0, 4.0],
                "pads": [],
                "current_xy_mm": [6.5, 7.5],
                "current_rot": 180,
                "fixed": False,
            },
            {"ref": "C1", "type": "capacitor", "size_mm": [2.0, 2.0], "pads": []},
        ],
        "graph": {"sequence": ["U1", "C1"]},
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")

    runtime_path, temp_path, meta = materialize_fixed_refs_task_json(str(path), ["U*"])
    try:
        data = json.loads(open(runtime_path, encoding="utf-8").read())
    finally:
        if temp_path:
            os.unlink(temp_path)

    u1 = next(c for c in data["components"] if c["ref"] == "U1")
    c1 = next(c for c in data["components"] if c["ref"] == "C1")
    assert u1["fixed"] is True
    assert u1["fixed_xy_mm"] == [6.5, 7.5]
    assert u1["fixed_rot"] == 180
    assert c1.get("fixed") is not True
    assert meta["matched"] == ["U1"]
    assert meta["unmatched"] == []


def test_kicad_parser_reads_locked_footprints():
    board = parse_kicad_pcb(
        r'''
        (kicad_pcb
          (net 1 "N1")
          (gr_rect (start 0 0) (end 20 20) (layer "Edge.Cuts") (width 0.1))
          (footprint "Device:R" (layer "F.Cu")
            (locked yes)
            (at 5 6 90)
            (property "Reference" "R1")
            (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "N1"))
          )
          (footprint "Device:C" (layer "F.Cu")
            (at 8 9 0)
            (property "Reference" "C1")
            (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "N1"))
          )
        )
        '''
    )
    by_ref = {fp.ref: fp for fp in board.footprints}
    assert by_ref["R1"].locked is True
    assert by_ref["C1"].locked is False


def test_step1_exports_locked_and_current_pose(tmp_path):
    from scripts.step1_kicad_to_json import build_task_from_kicad

    board_path = tmp_path / "locked.kicad_pcb"
    board_path.write_text(
        r'''
        (kicad_pcb
          (net 1 "N1")
          (gr_rect (start 0 0) (end 20 20) (layer "Edge.Cuts") (width 0.1))
          (footprint "Device:R" (layer "F.Cu")
            (locked yes)
            (at 5 6 90)
            (property "Reference" "R1")
            (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "N1"))
          )
          (footprint "Device:C" (layer "F.Cu")
            (at 8 9 0)
            (property "Reference" "C1")
            (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "N1"))
          )
        )
        ''',
        encoding="utf-8",
    )

    task = build_task_from_kicad(board_path, grid_mm=1.0)
    by_ref = {c["ref"]: c for c in task["components"]}
    assert by_ref["R1"]["fixed"] is True
    assert by_ref["R1"]["locked"] is True
    assert by_ref["R1"]["fixed_xy_mm"] == by_ref["R1"]["current_xy_mm"]
    assert by_ref["C1"]["fixed"] is False
    assert "current_xy_mm" in by_ref["C1"]
