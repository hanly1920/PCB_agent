from pcbplace.utils import (
    drop_mounting_holes_from_task_json,
    infer_type,
    is_mounting_hole_component,
)


def test_reference_and_footprint_mounting_hole_conventions():
    assert infer_type("H1", "") == "mech_mounting_hole"
    assert infer_type("MH1", "") == "mech_mounting_hole"
    assert infer_type("MH_4", "") == "mech_mounting_hole"
    assert infer_type("HOLE_M32", "HOLE_M3") == "mech_mounting_hole"
    assert infer_type("P1", "Mounting_Holes:MountingHole_2.7mm_M2.5") == "mech_mounting_hole"


def test_ref_star_star_requires_mechanical_evidence():
    assert infer_type("REF**", "MountingHole:MountingHole_3.2mm_M3") == "mech_mounting_hole"
    assert is_mounting_hole_component({
        "ref": "REF**",
        "type": "misc",
        "footprint": "MountingHole:MountingHole_3.2mm_M3",
    })

    assert infer_type("REF**", "6502Library:1pin") != "mech_mounting_hole"
    assert not is_mounting_hole_component({
        "ref": "REF**",
        "type": "misc",
        "footprint": "6502Library:1pin",
    })


def test_broad_h_and_screw_matches_do_not_delete_real_parts():
    assert infer_type("HAT1", "Module:HAT") != "mech_mounting_hole"
    assert infer_type(
        "RV1",
        "Potentiometers:Potentiometer_Bourns_3296W_3-8Zoll_Inline_ScrewUp",
    ) != "mech_mounting_hole"
    assert infer_type(
        "J1",
        "TerminalBlock:TerminalBlock_Screw_1x02_P5.08mm",
    ).startswith("conn_")


def test_mounting_hole_is_removed_from_task_and_graph_case_insensitively():
    data = {
        "components": [
            {
                "ref": "ref**",
                "type": "misc",
                "footprint": "MountingHole:MountingHole_3.2mm_M3",
                "pads": [],
            },
            {"ref": "U1", "type": "chip", "pads": []},
            {"ref": "J1", "type": "conn_header", "pads": []},
        ],
        "nets": {
            "N1": ["REF**.1", "U1.1", "J1.1"],
            "N2": ["ref**.2", "U1.2"],
        },
        "graph": {
            "sequence": ["REF**", "U1", "J1"],
            "nodes": ["ref**", "U1", "J1"],
        },
    }

    out = drop_mounting_holes_from_task_json(data)

    assert [c["ref"] for c in out["components"]] == ["U1", "J1"]
    assert out["nets"] == {"N1": ["U1.1", "J1.1"]}
    assert out["graph"]["sequence"] == ["U1", "J1"]
    assert out["graph"]["nodes"] == ["U1", "J1"]
    assert out["meta"]["ignored_mounting_holes"] == ["REF**"]


def test_duplicate_ref_does_not_delete_retained_electrical_component():
    data = {
        "components": [
            {
                "ref": "REF**",
                "type": "misc",
                "footprint": "MountingHole:MountingHole_3.2mm_M3",
            },
            {
                "ref": "REF**",
                "type": "misc",
                "footprint": "6502Library:1pin",
            },
            {"ref": "U1", "type": "chip"},
        ],
        "nets": {"N1": ["REF**.1", "U1.1"]},
        "graph": {"sequence": ["REF**", "U1"]},
    }

    out = drop_mounting_holes_from_task_json(data)

    assert [c.get("footprint", "") for c in out["components"]] == [
        "6502Library:1pin",
        "",
    ]
    assert out["nets"] == {"N1": ["REF**.1", "U1.1"]}
    assert out["graph"]["sequence"] == ["REF**", "U1"]
    assert out["meta"]["mounting_hole_ambiguous_refs_kept_in_graph"] == ["REF**"]
