from __future__ import annotations
import json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def seed_everything(seed: int) -> None:
    random.seed(seed)

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def bbox_from_points(points: List[Tuple[float,float]]) -> Tuple[float,float,float,float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def hpwl_from_pins(pin_xy: List[Tuple[float,float]]) -> float:
    if not pin_xy:
        return 0.0
    xs = [p[0] for p in pin_xy]
    ys = [p[1] for p in pin_xy]
    return (max(xs)-min(xs)) + (max(ys)-min(ys))

def nslw_from_pins(pin_xy: List[Tuple[float,float]]) -> float:
    """A lightweight proxy for NSLW.
    Here we use (hpwl * log(1+deg)) to upweight bigger nets, but keep it smooth.
    Replace with your exact NSLW definition if needed.
    """
    deg = len(pin_xy)
    if deg <= 1:
        return 0.0
    return hpwl_from_pins(pin_xy) * math.log(1.0 + deg)

def infer_type(ref: str, footprint: str = "") -> str:
    """Infer **fine-grained** component type.

    This value is used by:
      - sequence ordering heuristic
      - environment hard constraints / bias (edge placement)
      - json exports for later steps

    Downstream model tokens should use :func:`coarse_type_from_fine` to remain stable.
    """
    r = (ref or "").upper().strip()
    fp = (footprint or "").upper().strip()
    # Split KiCad footprint "Lib:Name" to avoid false matches (e.g., "TUSB..." / "Diode-SMA")
    fp_lib, fp_name = "", fp
    if ":" in fp:
        fp_lib, fp_name = fp.split(":", 1)

    # ----------------------------
    # Mechanical / mounting
    # ----------------------------
    if r.startswith(("MH", "H")) or any(k in fp for k in ["MOUNTINGHOLE", "MOUNTING_HOLE", "NPTH", "SCREW", "STANDOFF"]):
        return "mech_mounting_hole"
    if any(k in fp for k in ["SHIELD", "HEATSINK", "CASE", "BRACKET"]):
        return "mech_shield"

    # ----------------------------
    # UI / human interface (optional)
    # ----------------------------
    if r.startswith(("SW", "BTN", "KEY")) or any(k in fp for k in ["SWITCH", "BUTTON", "TACT", "PUSH"]):
        return "ui_button"
    if r.startswith("LED") or ("LED" in fp and r.startswith("D")):
        return "ui_led"

    # ----------------------------
    # Connectors / interfaces
    # ----------------------------
    is_conn_ref = r.startswith(("J", "P", "CN", "CON"))
    # Only treat as connector if footprint *library* looks like a connector library,
    # OR ref designator looks like a connector. This prevents "TUSB..." and "Diode-SMA" false positives.
    is_conn_lib = any(k in fp_lib for k in ["CONNECTOR", "CONN_","CONN-","CONN ","JST","TERMINALBLOCK"]) or fp_lib.startswith(("CONN", "CONNECTOR"))
    if is_conn_ref or is_conn_lib:
        # USB / Type-C
        if "USB" in fp_name:
            if any(k in fp_name for k in ["TYPE-C", "TYPEC", "USB_C", "USBC"]):
                return "conn_typec"
            return "conn_usb"
        # HDMI / DP
        if "HDMI" in fp_name:
            return "conn_hdmi"
        if any(k in fp_name for k in ["DISPLAYPORT", "DP_"]):
            return "conn_displayport"
        # Ethernet / RJ
        if any(k in fp_name for k in ["RJ45", "8P8C"]) or ("RJ45" in r):
            return "conn_rj45"
        if any(k in fp_name for k in ["RJ11", "6P"]) or ("RJ11" in r):
            return "conn_rj11"
        # RF
        if any(k in fp_name for k in ["SMA", "BNC", "U.FL", "UFL", "IPEX", "MMCX", "SMB"]):
            return "conn_rf"
        # Audio / barrel jack
        if any(k in fp_name for k in ["AUDIO", "PHONE_JACK", "JACK_3.5", "JACK_6.35", "TRRS", "TRS"]):
            return "conn_audio_jack"
        if any(k in fp_name for k in ["BARREL", "DC_JACK", "POWER_JACK"]):
            return "conn_barrel_jack"
        # Terminal block / screw
        if any(k in fp_name for k in ["TERMINAL", "SCREW_TERMINAL", "TERMINALBLOCK", "TBLOCK"]):
            return "conn_terminal_block"
        # FPC / FFC
        if any(k in fp_name for k in ["FPC", "FFC"]):
            return "conn_fpc"
        # SD / SIM
        if ("SD" in fp_name and "CARD" in fp_name) or any(k in fp_name for k in ["MICROSD", "TF_CARD"]):
            return "conn_sd_card"
        if "SIM" in fp_name:
            return "conn_sim"
        # JST / headers
        if "JST" in fp_name or "JST" in fp_lib:
            return "conn_jst"
        if any(k in fp_name for k in ["PINHEADER", "HEADER", "HDR", "PIN_HEADER"]):
            return "conn_pin_header"
        return "conn_other"

    # ----------------------------
    # ICs / semis / passives
    # ----------------------------
    if r.startswith("U") or any(k in fp for k in ["QFN", "BGA", "LQFP", "SOP", "SOIC", "TQFP", "QFP", "DFN"]):
        return "chip"
    if r.startswith("C"):
        return "capacitor"
    if r.startswith("R"):
        return "resistor"
    if r.startswith("L"):
        return "inductor"
    if r.startswith("D"):
        return "diode"
    if r.startswith("Q"):
        return "transistor"

    return "misc"


# -----------------------------------------------------------------------------
# Type helpers used across sequence/env/train.
# -----------------------------------------------------------------------------

_EDGE_REQUIRED_TYPES = {
    # high-confidence edge connectors
    "conn_usb",
    "conn_typec",
    "conn_hdmi",
    "conn_displayport",
    "conn_rj45",
    "conn_rj11",
    "conn_rf",
    "conn_audio_jack",
    "conn_barrel_jack",
    "conn_terminal_block",
    "conn_sd_card",
    "conn_sim",
}

_EDGE_PREFERRED_TYPES = {
    # all connectors generally prefer boundary, but only some are hard-required
    "conn_usb",
    "conn_typec",
    "conn_hdmi",
    "conn_displayport",
    "conn_rj45",
    "conn_rj11",
    "conn_rf",
    "conn_audio_jack",
    "conn_barrel_jack",
    "conn_terminal_block",
    "conn_sd_card",
    "conn_sim",
    "conn_fpc",
    "conn_jst",
    "conn_pin_header",
    "conn_other",
}

_FIXED_FIRST_TYPES = {
    "mech_mounting_hole",
    "mech_shield",
}

def coarse_type_from_fine(t: str) -> str:
    """Map fine type -> coarse type used by model tokens and legacy heuristics."""
    tt = (t or "").lower()
    if tt == "interface":
        return "interface"
    if tt.startswith("conn_"):
        return "interface"
    if tt == "mechanical":
        return "mechanical"
    if tt.startswith("mech_"):
        return "mechanical"
    if tt in ("ui_button", "ui_led"):
        return "misc"
    if tt in ("chip", "capacitor", "resistor", "inductor", "diode", "transistor", "misc"):
        return tt
    return "misc"

def is_connector_type(t: str) -> bool:
    tt = (t or "").lower()
    return tt.startswith("conn_") or tt == "interface"

def is_pad_bbox_only_type(t: str) -> bool:
    # For connectors/interfaces we only trust pad geometry for bbox/size.
    return is_connector_type(t)

def is_edge_required_type(t: str) -> bool:
    tt = (t or "").lower()
    return tt == "interface" or tt in _EDGE_REQUIRED_TYPES

def is_edge_preferred_type(t: str) -> bool:
    tt = (t or "").lower()
    return tt == "interface" or tt in _EDGE_PREFERRED_TYPES

def is_fixed_first_type(t: str, area_mm2: float | None = None) -> bool:
    tt = (t or "").lower()
    if tt in _FIXED_FIRST_TYPES:
        return True
    # Large parts (connectors are handled separately by edge groups)
    if area_mm2 is not None and float(area_mm2) >= 80.0:
        # heuristic: >= 80 mm^2 (~9x9mm)
        return True
    return False
