from __future__ import annotations
import copy, json, math, random, re
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

def w_hpwl_from_pins(pin_xy: List[Tuple[float,float]]) -> float:
    """Weighted HPWL proxy: HPWL * log(1 + pin_count).

    Earlier versions of this project called this metric ``nslw``.  That was
    misleading: it is a fanout-weighted HPWL term, not the PCBAgent NSLW
    metric.  The true NSLW is implemented in ``PlacementEnv`` as a count of
    surface-layer wires.
    """
    deg = len(pin_xy)
    if deg <= 1:
        return 0.0
    return hpwl_from_pins(pin_xy) * math.log(1.0 + deg)


def nslw_from_pins(pin_xy: List[Tuple[float,float]]) -> float:
    """Backward-compatible alias for old checkpoints/scripts.

    Prefer :func:`w_hpwl_from_pins` in new code.
    """
    return w_hpwl_from_pins(pin_xy)


def _normalize_component_ref(ref: Any) -> str:
    """Normalize a reference designator for case-insensitive comparisons."""
    return str(ref or "").strip().upper()


def _has_mounting_hole_ref_evidence(ref: Any) -> bool:
    """Recognize dedicated mounting-hole reference conventions.

    Deliberately avoids a broad ``startswith("H")`` rule, which would also
    swallow legitimate references such as HAT1 or HDRV1.
    """
    r = _normalize_component_ref(ref)
    return bool(
        re.fullmatch(r"H\d+", r)
        or re.fullmatch(r"MH(?:[_-]?\d+)?", r)
        or re.fullmatch(r"HOLE(?:[_-].*|\d*)", r)
        or re.fullmatch(r"MOUNT(?:ING)?[_-]?HOLE(?:[_-]?\d+)?", r)
    )


def _has_mounting_hole_text_evidence(value: Any) -> bool:
    """Recognize a standalone mounting-hole footprint/value description.

    The checks are intentionally narrower than matching the words ``SCREW`` or
    ``NPTH`` anywhere. Connector and potentiometer footprints may contain those
    words while still being electrical components.
    """
    text = str(value or "").strip().upper()
    if not text:
        return False

    compact = re.sub(r"[^A-Z0-9]+", "", text)
    basename = text.rsplit(":", 1)[-1].strip()
    basename_compact = re.sub(r"[^A-Z0-9]+", "", basename)

    if "MOUNTINGHOLE" in compact:
        return True
    if "STANDOFF" in compact or "SCREWHOLE" in compact:
        return True

    # Custom names seen in PCB datasets: HOLE_M3, 3MM_HOLE, NPTH_3MM.
    if re.match(r"^HOLE(?:[_-]|$)", basename):
        return True
    if re.match(r"^\d+(?:\.\d+)?MM[_-]?HOLE(?:[_-]|$)", basename):
        return True
    if re.match(r"^M\d+(?:\.\d+)?[_-]?HOLE(?:[_-]|$)", basename):
        return True
    if re.match(r"^NPTH(?:[_-]|$)", basename):
        return True

    # Compact variants such as HOLEM3 and 3MMHOLE.
    if re.match(r"^HOLEM\d", basename_compact):
        return True
    if re.match(r"^\d+(?:\d+)?MMHOLE", basename_compact):
        return True

    return False


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
    # REF** is KiCad's generic unannotated reference, not a mounting-hole type
    # by itself. Treat it as a hole only when the footprint/value supplies
    # mechanical evidence. This preserves legitimate unannotated one-pin parts.
    if _has_mounting_hole_ref_evidence(r) or _has_mounting_hole_text_evidence(fp):
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

def is_mounting_hole_type(t: str) -> bool:
    """Return True for component types that must be invisible to train/infer.

    Mounting holes are board/mechanical constraints, not placeable electrical
    components.  They must not become model tokens, action targets, graph nodes,
    or metric references.
    """
    return str(t or "").lower().strip() == "mech_mounting_hole"


def is_mounting_hole_component(c: Dict[str, Any]) -> bool:
    """Identify a mounting-hole component without relying on REF** alone.

    New JSON normally carries ``type="mech_mounting_hole"``. For legacy JSON,
    dedicated reference conventions and mechanical footprint/value text are
    accepted as evidence.
    """
    if not isinstance(c, dict):
        return False

    typ = str(c.get("type", "") or "").lower().strip()
    if is_mounting_hole_type(typ):
        return True

    if _has_mounting_hole_ref_evidence(c.get("ref")):
        return True

    for key in (
        "footprint",
        "value",
        "name",
        "description",
        "lib_id",
        "package",
    ):
        if _has_mounting_hole_text_evidence(c.get(key)):
            return True

    return False


def _pin_ref_for_filter(pin: Any) -> str:
    """Extract and normalize a component ref from a net/graph token."""
    if isinstance(pin, dict):
        val = (
            pin.get("ref")
            or pin.get("component")
            or pin.get("component_ref")
            or pin.get("refdes")
            or pin.get("node")
            or pin.get("pin")
            or ""
        )
    elif isinstance(pin, (list, tuple)) and pin:
        val = pin[0]
    else:
        val = pin

    s = str(val or "").strip()
    if not s:
        return ""

    # REF** contains asterisks outside the normal refdes character set.
    match = re.match(r"^(REF\*\*(?:[_-]\d+)?)(?:[.:/].*)?$", s, re.IGNORECASE)
    if match:
        return _normalize_component_ref(match.group(1))

    # Strip pad suffix from U1.3 / U1:3 / U1/3 while preserving refs like USB1.
    match = re.match(r"^([A-Za-z]+[A-Za-z0-9_*_-]*?)(?:[.:/].*)?$", s)
    return _normalize_component_ref(match.group(1) if match else s)


def _filter_ref_list_for_mounting_holes(value: Any, ignored_refs: set[str]) -> Any:
    """Filter a list/dict/scalar field containing component refs.

    Returns the same broad container type where practical.
    """
    if value is None:
        return value

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            ref = _pin_ref_for_filter(v if v not in (None, "") else k)
            if ref not in ignored_refs:
                out[k] = v
        return out

    if isinstance(value, list):
        out = []
        for item in value:
            ref = _pin_ref_for_filter(item)
            if ref not in ignored_refs:
                out.append(item)
        return out

    if isinstance(value, tuple):
        out = []
        for item in value:
            ref = _pin_ref_for_filter(item)
            if ref not in ignored_refs:
                out.append(item)
        return tuple(out)

    ref = _pin_ref_for_filter(value)
    return value if ref not in ignored_refs else None


def drop_mounting_holes_from_task_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove mounting holes from task-like JSON before train/infer.

    The filter removes mounting holes from:
      - components
      - nets / graph-like pin lists
      - graph.sequence and related sequence details
      - module/member reference lists
      - optional prediction/placement maps keyed by ref

    It intentionally does not modify board.bbox_mm: board outline still belongs
    to the physical PCB even if mechanical holes are ignored by the model.
    """
    if not isinstance(data, dict) or "components" not in data:
        return data

    out = copy.deepcopy(data)
    components = [c for c in (out.get("components") or []) if isinstance(c, dict)]
    mounting_holes = [c for c in components if is_mounting_hole_component(c)]
    if not mounting_holes:
        return out

    kept_components = [c for c in components if not is_mounting_hole_component(c)]
    out["components"] = kept_components

    hole_refs = {
        _normalize_component_ref(c.get("ref"))
        for c in mounting_holes
        if _normalize_component_ref(c.get("ref"))
    }
    kept_refs = {
        _normalize_component_ref(c.get("ref"))
        for c in kept_components
        if _normalize_component_ref(c.get("ref"))
    }

    # A legacy file can contain duplicate unannotated refs such as REF**.
    # Remove graph/net entries only when no retained component shares that ref;
    # otherwise the endpoint is ambiguous and deleting it could corrupt a real
    # electrical component.
    ignored_refs = hole_refs - kept_refs

    # Remove mounting-hole pads from top-level nets.  Drop degenerate nets with
    # fewer than two remaining endpoints because they do not define connectivity.
    nets = out.get("nets")
    if isinstance(nets, dict):
        new_nets: Dict[str, Any] = {}
        for net, pins in nets.items():
            kept = _filter_ref_list_for_mounting_holes(pins, ignored_refs)
            if isinstance(kept, dict):
                keep_count = len(kept)
            elif isinstance(kept, (list, tuple, set)):
                keep_count = len(kept)
            elif kept is None:
                keep_count = 0
            else:
                keep_count = 1
            if keep_count >= 2:
                new_nets[net] = kept
        out["nets"] = new_nets

    # Filter graph fields that may list component refs or net endpoints.
    graph = out.get("graph")
    if isinstance(graph, dict):
        for key in (
            "sequence",
            "module_sequence",
            "fixed_sequence",
            "placement_sequence",
            "nodes",
            "components",
            "refs",
        ):
            if key in graph:
                graph[key] = _filter_ref_list_for_mounting_holes(graph.get(key), ignored_refs)

        if isinstance(graph.get("module_sequence_detail"), list):
            graph["module_sequence_detail"] = [
                item for item in graph["module_sequence_detail"]
                if _pin_ref_for_filter(item.get("ref") if isinstance(item, dict) else item) not in ignored_refs
            ]

        for key in ("nets", "edges", "hyperedges"):
            if isinstance(graph.get(key), dict):
                graph[key] = {
                    net: kept
                    for net, value in graph[key].items()
                    for kept in [_filter_ref_list_for_mounting_holes(value, ignored_refs)]
                    if (len(kept) if isinstance(kept, (list, tuple, set, dict)) else int(kept is not None)) >= 2
                }
        out["graph"] = graph

    # Filter modules or annotations that may carry member refs.
    def _filter_module_container(container: Any) -> Any:
        if isinstance(container, list):
            new_items = []
            for item in container:
                if isinstance(item, dict):
                    item = copy.deepcopy(item)
                    for key in (
                        "refs", "components", "members", "nodes",
                        "member_refs", "component_refs", "sequence",
                    ):
                        if key in item:
                            item[key] = _filter_ref_list_for_mounting_holes(item.get(key), ignored_refs)
                    # Drop empty modules that contain no remaining members.
                    member_fields = [
                        item.get(k) for k in ("refs", "components", "members", "nodes", "member_refs", "component_refs")
                        if k in item
                    ]
                    if member_fields:
                        sizes = [
                            len(v) if isinstance(v, (list, tuple, set, dict)) else int(v is not None)
                            for v in member_fields
                        ]
                        if max(sizes or [0]) <= 0:
                            continue
                new_items.append(item)
            return new_items
        if isinstance(container, dict):
            return {
                k: _filter_module_container(v)
                for k, v in container.items()
                if _pin_ref_for_filter(k) not in ignored_refs
            }
        return container

    for key in ("modules", "module_annotations", "semantic_modules"):
        if key in out:
            out[key] = _filter_module_container(out.get(key))

    # Remove optional placement/prediction entries keyed by ref.
    for key in ("placed", "placements", "fixed", "fixed_components", "locked_components"):
        val = out.get(key)
        if isinstance(val, dict):
            out[key] = {k: v for k, v in val.items() if _normalize_component_ref(k) not in ignored_refs}
        elif isinstance(val, list):
            out[key] = [
                item for item in val
                if _pin_ref_for_filter(item.get("ref") if isinstance(item, dict) else item) not in ignored_refs
            ]

    meta = out.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["ignored_mounting_holes"] = sorted(hole_refs)
        meta["mounting_hole_refs_removed_from_graph"] = sorted(ignored_refs)
        meta["mounting_hole_ambiguous_refs_kept_in_graph"] = sorted(
            hole_refs & kept_refs
        )
        meta["mounting_holes_ignored_for_model"] = True

    return out

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
