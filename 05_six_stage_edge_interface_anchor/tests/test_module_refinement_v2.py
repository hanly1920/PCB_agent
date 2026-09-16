from __future__ import annotations

from pcbplace.env import Component
from pcbplace.module_partition import infer_modules
from pcbplace.module_region_proposer import attach_prior_and_expert_regions


def _comp(ref: str, typ: str, size=(3.0, 2.0), pads=(), **kwargs):
    return Component(ref=ref, type=typ, size_mm=size, pads=list(pads), allowed_sides=[], **kwargs)


def test_connector_does_not_absorb_relay_or_core_anchor():
    comps = [
        _comp("J1", "conn_usb", pads=[("USB_DP", (0, 0)), ("USB_DM", (0, 0))], semantic_class="interface", side_preference="edge_top", must_touch_boundary=True),
        _comp("R1", "resistor", pads=[("USB_DP", (0, 0)), ("MCU_DP", (0, 0))], semantic_class="interface_support"),
        _comp("K1", "misc", size=(62.0, 17.8), pads=[("VIN", (0, 0)), ("LOAD", (0, 0))], semantic_class="power"),
        _comp("U1", "chip", size=(8.0, 8.0), pads=[("MCU_DP", (0, 0)), ("GPIO", (0, 0))], semantic_class="core"),
    ]
    modules = infer_modules(comps, board_bbox=(0.0, 0.0, 80.0, 50.0), grid_mm=1.0)
    interface = next(m for m in modules if m["anchor_ref"] == "J1")
    assert "K1" not in interface["members"]
    assert "U1" not in interface["members"]
    relay = next(m for m in modules if "K1" in m["members"])
    core = next(m for m in modules if "U1" in m["members"])
    assert relay["module_type"] == "power_or_driver"
    assert core["module_type"] == "core_ic"


def test_unconstrained_interface_prior_is_broad_and_low_confidence():
    comps = [
        _comp("J1", "conn_other", pads=[("SIG", (0, 0))], semantic_class="interface"),
    ]
    modules = infer_modules(comps, board_bbox=(0.0, 0.0, 100.0, 80.0), grid_mm=1.0)
    out = attach_prior_and_expert_regions(modules, {c.ref: c for c in comps}, (0.0, 0.0, 100.0, 80.0))
    prior = out[0]["prior_region"]
    assert prior["source"] == "rule_v2_interface"
    assert prior["confidence"] <= 0.30
    assert prior["bbox_mm"] == [8.0, 6.4, 92.0, 73.60000000000001]
