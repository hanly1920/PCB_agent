from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set, Tuple

from .utils import infer_type, is_connector_type, is_edge_required_type

POWER_KEYS = ("VCC", "VDD", "3V3", "5V", "1V8", "VSYS", "VBUS", "VIN", "VOUT", "VDDA", "VSSA", "AVDD", "DVDD", "GND")

EDGE_REQUIRED_CONN_TYPES = {
    "conn_typec", "conn_usb", "conn_hdmi", "conn_displayport", "conn_rj45", "conn_rj11", "conn_rf",
    "conn_audio_jack", "conn_barrel_jack", "conn_terminal_block", "conn_sd_card", "conn_sim", "interface",
}

UI_TYPES = {"ui_button", "ui_led", "ui_switch", "ui_buzzer", "ui_display", "ui_encoder"}
MECH_TYPES = {"mechanical", "mech_mounting_hole", "mech_shield", "mech_heatsink", "mech_slot"}
BULKY_TYPES = {"inductor", "transformer", "relay", "fuse", "connector_housing"}
BULKY_AREA_MM2 = 50.0


def _is_power(net: str) -> bool:
    u = str(net).upper()
    return any(k in u for k in POWER_KEYS)


def _is_gnd(net: str) -> bool:
    u = str(net).upper().strip()
    return ("GND" in u) or (u in ("0", "0V", "AGND", "DGND", "PGND", "VSSA"))


def _safe_area_mm2(comp: Dict[str, Any]) -> float:
    w, h = comp.get("size_mm", [1.0, 1.0])
    try:
        return max(1e-3, float(w) * float(h))
    except Exception:
        return 1.0


def _text_blob(ref: str, comp: Dict[str, Any]) -> str:
    return f"{ref} {comp.get('footprint', '')}".upper()


def _is_memory_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    return any(k in s for k in ("DDR", "SDRAM", "DRAM", "RAM", "FLASH", "EEPROM", "QSPI", "NOR", "NAND", "EMMC", "PSRAM"))


def _is_clock_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    return any(k in s for k in ("XTAL", "CRYSTAL", "OSC", "OSCILLATOR", "CLK"))


def _is_power_ic_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    return any(k in s for k in ("PMIC", "REG", "LDO", "BUCK", "BOOST", "DCDC", "DCDC", "CHARGER", "POWER"))


def _is_interface_support_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    t = str(comp.get("type") or infer_type(ref, comp.get("footprint", ""))).lower()
    return (
        any(k in s for k in ("ESD", "TVS", "CMCHOKE", "COMMONMODE", "COMMON_MODE", "CHOKE", "FILTER", "FERRITE", "MAGNETICS", "LAN", "ETH", "USBLC", "POLYFUSE", "FUSE"))
        or t in {"diode", "inductor"}
    )


def _is_bridge_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    return any(k in s for k in ("LEVEL", "SHIFT", "TRANSCEIVER", "BUFFER", "DRV", "DRIVER", "MUX", "DEMUX", "SWITCH", "ISOLATOR", "OPTO", "PHY", "REPEATER", "TRANSLATOR"))


def _is_sensor_like(ref: str, comp: Dict[str, Any]) -> bool:
    s = _text_blob(ref, comp)
    return any(k in s for k in ("SENSOR", "IMU", "ACCEL", "GYRO", "TEMP", "PRESS", "HALL", "TOUCH", "MIC", "CAM"))


def _semantic_value(comp: Dict[str, Any], key: str, default: str = "") -> str:
    semantic = comp.get("semantic") or {}
    return str(comp.get(key) or semantic.get(key) or default).strip().lower()


def _semantic_review_status(comp: Dict[str, Any]) -> str:
    review = comp.get("semantic_review") or {}
    return str(review.get("review_status") or "seeded").strip().lower()



def _net_weight(net: str, fanout: int) -> float:
    if _is_gnd(net):
        return 0.0
    base = 1.0 / math.sqrt(max(1.0, float(max(2, fanout)) - 1.0))
    if _is_power(net):
        base *= 0.65
    return float(base)


class _GraphBundle:
    def __init__(self, task: Dict[str, Any]):
        comps = task.get("components", [])
        nets = task.get("nets", {})
        self.refs = [c["ref"] for c in comps]
        self.comp_by_ref = {c["ref"]: c for c in comps}

        pin_to_net: Dict[str, str] = {}
        for net, pins in nets.items():
            for p in pins:
                pin_to_net[p] = str(net)

        self.nets_of: Dict[str, Set[str]] = {c["ref"]: set() for c in comps}
        for c in comps:
            ref = c["ref"]
            for pad in c.get("pads", []):
                n = pad.get("net")
                if n:
                    self.nets_of[ref].add(str(n))
            for pin, net in pin_to_net.items():
                if pin.startswith(ref + "."):
                    self.nets_of[ref].add(str(net))

        self.net_to_refs: Dict[str, Set[str]] = defaultdict(set)
        for ref, ns in self.nets_of.items():
            for n in ns:
                if not _is_gnd(n):
                    self.net_to_refs[n].add(ref)

        self.net_fanout: Dict[str, int] = {n: len(rs) for n, rs in self.net_to_refs.items()}

        self.adj: Dict[str, Set[str]] = {r: set() for r in self.refs}
        self.edge_w: Dict[Tuple[str, str], float] = defaultdict(float)
        self.edge_nets: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for net, rs in self.net_to_refs.items():
            rs_list = list(rs)
            wt = _net_weight(net, self.net_fanout[net])
            if wt <= 0.0:
                continue
            for i in range(len(rs_list)):
                for j in range(i + 1, len(rs_list)):
                    a, b = rs_list[i], rs_list[j]
                    if a > b:
                        a, b = b, a
                    self.adj[a].add(b)
                    self.adj[b].add(a)
                    self.edge_w[(a, b)] += wt
                    self.edge_nets[(a, b)].add(net)

        self.degree: Dict[str, int] = {r: len(self.adj[r]) for r in self.refs}
        self.weighted_degree: Dict[str, float] = {r: 0.0 for r in self.refs}
        for (a, b), wt in self.edge_w.items():
            self.weighted_degree[a] += wt
            self.weighted_degree[b] += wt

        self.neighbor_w: Dict[str, Dict[str, float]] = {r: {} for r in self.refs}
        for (a, b), wt in self.edge_w.items():
            self.neighbor_w[a][b] = wt
            self.neighbor_w[b][a] = wt

    def pair_weight(self, a: str, b: str) -> float:
        x, y = (a, b) if a < b else (b, a)
        return float(self.edge_w.get((x, y), 0.0))

    def pair_nets(self, a: str, b: str) -> Set[str]:
        x, y = (a, b) if a < b else (b, a)
        return set(self.edge_nets.get((x, y), set()))


def _multi_source_hops(adj: Dict[str, Set[str]], seeds: Iterable[str], max_depth: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    q = deque()
    for s in seeds:
        if s not in out:
            out[s] = 0
            q.append(s)
    while q:
        cur = q.popleft()
        d = out[cur]
        if d >= max_depth:
            continue
        for nb in adj.get(cur, set()):
            if nb not in out:
                out[nb] = d + 1
                q.append(nb)
    return out


def _norm_map(vals: Dict[str, float]) -> Dict[str, float]:
    if not vals:
        return {}
    lo = min(vals.values())
    hi = max(vals.values())
    if hi - lo <= 1e-9:
        return {k: 0.0 for k in vals}
    return {k: (float(v) - lo) / (hi - lo) for k, v in vals.items()}


def _sorted_unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def generate_sequence(task: Dict[str, Any], bfs_depth: int = 3, return_meta: bool = False):
    comps = task.get("components", [])
    if not comps:
        return ([], {}) if return_meta else []

    g = _GraphBundle(task)
    refs = list(g.refs)
    comp_by_ref = g.comp_by_ref
    nets_of = g.nets_of

    def ctype(ref: str) -> str:
        c = comp_by_ref[ref]
        return str(c.get("type") or infer_type(ref, c.get("footprint", ""))).lower()

    def area(ref: str) -> float:
        return _safe_area_mm2(comp_by_ref[ref])
    def semantic_class(ref: str) -> str:
        return _semantic_value(comp_by_ref[ref], "semantic_class", "other")

    def functional_group(ref: str) -> str:
        return _semantic_value(comp_by_ref[ref], "functional_group", "misc")

    def anchor_ref(ref: str) -> str:
        return _semantic_value(comp_by_ref[ref], "anchor_ref", "")

    def review_status(ref: str) -> str:
        return _semantic_review_status(comp_by_ref[ref])


    def sqrt_area(ref: str) -> float:
        return math.sqrt(max(1e-6, area(ref)))

    def has_allowed_sides(ref: str) -> bool:
        return bool(comp_by_ref[ref].get("allowed_sides"))

    def must_touch_boundary(ref: str) -> bool:
        comp = comp_by_ref[ref]
        explicit = comp.get("must_touch_boundary", None)
        if explicit is None:
            semantic = comp.get("semantic") or {}
            explicit = semantic.get("must_touch_boundary", None)
        if explicit is not None:
            return bool(explicit)

        if is_edge_required_type(ctype(ref)):
            return True

        semantic = comp.get("semantic") or {}
        semantic_class = str(
            semantic.get("semantic_class", comp.get("semantic_class", "")) or ""
        ).lower()
        if semantic_class == "mechanical_edge_interface":
            return True

        return False

    def is_conn(ref: str) -> bool:
        return is_connector_type(ctype(ref))

    def is_mech(ref: str) -> bool:
        t = ctype(ref)
        return t in MECH_TYPES or t.startswith("mech_")

    def is_ui(ref: str) -> bool:
        return ctype(ref) in UI_TYPES

    def is_bulky(ref: str) -> bool:
        return ctype(ref) in BULKY_TYPES or area(ref) >= BULKY_AREA_MM2

    def is_chip(ref: str) -> bool:
        return ctype(ref) == "chip"

    def is_cap(ref: str) -> bool:
        return ctype(ref) == "capacitor"

    def is_res(ref: str) -> bool:
        return ctype(ref) == "resistor"

    def is_ind(ref: str) -> bool:
        return ctype(ref) == "inductor"

    def is_diode(ref: str) -> bool:
        return ctype(ref) == "diode"

    def is_transistor(ref: str) -> bool:
        return ctype(ref) == "transistor"

    def is_memory(ref: str) -> bool:
        return _is_memory_like(ref, comp_by_ref[ref])

    def is_clock(ref: str) -> bool:
        return _is_clock_like(ref, comp_by_ref[ref])

    def is_power_ic(ref: str) -> bool:
        return _is_power_ic_like(ref, comp_by_ref[ref])

    def is_interface_support(ref: str) -> bool:
        return _is_interface_support_like(ref, comp_by_ref[ref])

    def is_bridge(ref: str) -> bool:
        return _is_bridge_like(ref, comp_by_ref[ref])

    def is_sensor(ref: str) -> bool:
        return _is_sensor_like(ref, comp_by_ref[ref])

    def power_net_count(ref: str) -> int:
        return sum(1 for n in nets_of[ref] if _is_power(n) and not _is_gnd(n))

    def signal_net_count(ref: str) -> int:
        return sum(1 for n in nets_of[ref] if (not _is_gnd(n)) and (not _is_power(n)))

    def is_decap(ref: str) -> bool:
        if not is_cap(ref):
            return False
        nets = nets_of[ref]
        has_gnd = any(_is_gnd(n) for n in nets)
        has_pwr = any(_is_power(n) and not _is_gnd(n) for n in nets)
        return has_gnd and has_pwr

    def is_small_passive(ref: str) -> bool:
        return (is_cap(ref) or is_res(ref) or is_ind(ref)) and area(ref) <= 12.0 and g.degree.get(ref, 0) <= 3

    area_n = _norm_map({r: area(r) for r in refs})
    sqrt_area_n = _norm_map({r: sqrt_area(r) for r in refs})
    degree_n = _norm_map({r: float(g.degree.get(r, 0)) for r in refs})
    wdeg_n = _norm_map({r: float(g.weighted_degree.get(r, 0.0)) for r in refs})
    power_n = _norm_map({r: float(power_net_count(r)) for r in refs})
    signal_n = _norm_map({r: float(signal_net_count(r)) for r in refs})

    chipish_refs = [r for r in refs if is_chip(r) or is_memory(r) or is_clock(r) or is_power_ic(r) or is_sensor(r)]

    chip_neighbor_strength: Dict[str, float] = {}
    for r in refs:
        chip_neighbor_strength[r] = sum(g.pair_weight(r, nb) for nb in chipish_refs if nb != r)
    chip_neighbor_n = _norm_map(chip_neighbor_strength)

    hard_edge_anchors = [
        r for r in refs
        if must_touch_boundary(r) or is_mech(r) or semantic_class(r) in {"interface", "mechanical_edge_interface", "ui"}
    ]
    hard_edge_anchors = _sorted_unique(sorted(hard_edge_anchors, key=lambda r: (
        -int(review_status(r) in {"approved", "edited", "manual_reviewed", "accepted", "confirmed"} and semantic_class(r) in {"interface", "mechanical_edge_interface", "ui"}),
        -int(must_touch_boundary(r)),
        -int(semantic_class(r) in {"interface", "mechanical_edge_interface"}),
        -int(is_edge_required_type(ctype(r))),
        -int(is_mech(r)),
        -area_n[r],
        -wdeg_n[r],
        r,
    )))
    hard_edge_anchor_set = set(hard_edge_anchors)

    edge_seed_conn_raw = {r: sum(g.pair_weight(r, s) for s in hard_edge_anchor_set if s != r) for r in refs}
    edge_seed_conn_n = _norm_map(edge_seed_conn_raw)

    core_seed_score: Dict[str, float] = {}
    for r in refs:
        score = 0.0
        score += 2.6 * float(is_chip(r))
        score += 2.0 * float(is_memory(r) or is_clock(r) or is_power_ic(r))
        score += 1.2 * float(is_sensor(r))
        score += 1.7 * wdeg_n[r]
        score += 1.2 * degree_n[r]
        score += 1.0 * power_n[r]
        score += 0.8 * signal_n[r]
        score += 0.9 * chip_neighbor_n[r]
        score += 0.8 * area_n[r]
        score -= 3.0 * float(r in hard_edge_anchor_set)
        score -= 1.7 * float(is_conn(r))
        score -= 1.4 * float(is_mech(r))
        score -= 1.0 * float(is_ui(r))
        score -= 1.0 * float(is_interface_support(r))
        score -= 1.1 * float(is_decap(r))
        score -= 0.7 * float(is_small_passive(r))
        core_seed_score[r] = score

    edge_anchor_score: Dict[str, float] = {}
    for r in refs:
        score = 0.0
        score += 3.0 * float(r in hard_edge_anchor_set)
        score += 2.2 * float(is_conn(r))
        score += 2.0 * float(is_mech(r))
        score += 1.4 * float(is_ui(r))
        score += 1.1 * float(is_bulky(r))
        score += 1.0 * edge_seed_conn_n[r]
        score += 0.7 * area_n[r]
        score += 0.5 * wdeg_n[r]
        score -= 1.6 * float(is_chip(r) or is_memory(r) or is_clock(r) or is_power_ic(r))
        score -= 1.0 * float(is_interface_support(r))
        score -= 0.8 * float(is_decap(r))
        edge_anchor_score[r] = score

    edge_soft_pool = [
        r for r in refs
        if r not in hard_edge_anchor_set
        and (is_conn(r) or is_mech(r) or is_ui(r) or is_bulky(r))
        and not (is_chip(r) or is_memory(r) or is_clock(r) or is_power_ic(r))
    ]
    edge_soft_pool.sort(key=lambda r: (-edge_anchor_score[r], r))
    max_edge_soft = max(0, min(6, int(math.ceil(0.10 * len(refs)))))
    edge_soft_anchors: List[str] = []
    for r in edge_soft_pool:
        if len(edge_soft_anchors) >= max_edge_soft:
            break
        existing_edge = list(hard_edge_anchor_set) + edge_soft_anchors
        redundancy = max((g.pair_weight(r, a) for a in existing_edge), default=0.0)
        adjusted = edge_anchor_score[r] - 0.55 * redundancy
        if adjusted >= 1.25:
            edge_soft_anchors.append(r)

    edge_anchors = _sorted_unique(hard_edge_anchors + edge_soft_anchors)
    edge_anchor_set = set(edge_anchors)

    core_candidate_pool = [
        r for r in refs
        if r not in edge_anchor_set
        and (is_chip(r) or is_memory(r) or is_clock(r) or is_power_ic(r) or wdeg_n[r] >= 0.35 or semantic_class(r) in {"core", "power"} or bool(anchor_ref(r)))
        and not is_decap(r)
        and not (is_small_passive(r) and not is_chip(r))
    ]
    core_candidate_pool.sort(key=lambda r: (-core_seed_score[r], r))
    max_core_anchors = max(1, min(8, int(math.ceil(0.14 * len(refs)))))
    core_anchors: List[str] = []
    for r in core_candidate_pool:
        redundancy = max((g.pair_weight(r, a) for a in core_anchors), default=0.0)
        edge_overlap_pen = 0.35 * edge_seed_conn_n[r]
        adjusted = core_seed_score[r] - 0.65 * redundancy - edge_overlap_pen
        if not core_anchors:
            core_anchors.append(r)
            continue
        if len(core_anchors) >= max_core_anchors:
            break
        if adjusted >= max(0.25, 0.45 * core_seed_score[core_anchors[0]]):
            core_anchors.append(r)
    if not core_anchors and core_candidate_pool:
        core_anchors = [core_candidate_pool[0]]
    core_anchor_set = set(_sorted_unique(core_anchors)) - edge_anchor_set
    core_anchors = [r for r in core_anchors if r in core_anchor_set]
    if not core_anchors and core_candidate_pool:
        fallback = next((r for r in core_candidate_pool if r not in edge_anchor_set), None)
        if fallback is not None:
            core_anchors = [fallback]
            core_anchor_set = {fallback}

    core_seed_conn_raw = {r: sum(g.pair_weight(r, s) for s in core_anchor_set if s != r) for r in refs}
    core_seed_conn_n = _norm_map(core_seed_conn_raw)

    edge_hops = _multi_source_hops(g.adj, edge_anchor_set, max_depth=max(1, bfs_depth)) if edge_anchor_set else {}
    core_hops = _multi_source_hops(g.adj, core_anchor_set, max_depth=max(1, bfs_depth)) if core_anchor_set else {}

    def hop_bonus(hops: Dict[str, int], ref: str) -> float:
        if ref not in hops:
            return 0.0
        d = int(hops[ref])
        if d <= 0:
            return 1.0
        return max(0.0, 1.0 - (float(d) / float(max(1, bfs_depth) + 0.5)))

    def conn_to_set_weight(ref: str, others: Set[str]) -> float:
        if not others:
            return 0.0
        return sum(g.neighbor_w.get(ref, {}).get(nb, 0.0) for nb in others if nb != ref)

    def support_core_bonus(ref: str) -> float:
        score = 0.0
        score += 1.6 * float(is_decap(ref))
        score += 1.2 * float(is_clock(ref))
        score += 1.2 * float(is_memory(ref))
        score += 1.0 * float(is_power_ic(ref))
        score += 0.8 * float(is_chip(ref))
        score += 0.5 * float(is_sensor(ref))
        if is_decap(ref):
            for nb in g.adj.get(ref, set()):
                if nb in core_anchor_set and any(_is_power(n) for n in g.pair_nets(ref, nb)):
                    score += 0.8
                    break
        return score

    def support_edge_bonus(ref: str) -> float:
        score = 0.0
        score += 1.6 * float(is_interface_support(ref))
        score += 1.2 * float(is_conn(ref))
        score += 0.9 * float(is_ui(ref))
        score += 0.8 * float(is_bulky(ref))
        if is_interface_support(ref):
            for nb in g.adj.get(ref, set()):
                if nb in edge_anchor_set:
                    score += 0.8
                    break
        return score

    zone_scores: Dict[str, Dict[str, float]] = {}
    zone_label: Dict[str, str] = {}
    for r in refs:
        edge_score = 0.0
        core_score = 0.0
        mid_score = 0.0

        edge_score += 3.2 * float(r in edge_anchor_set)
        edge_score += 2.0 * float(is_conn(r))
        edge_score += 1.8 * float(is_mech(r))
        edge_score += 1.4 * float(is_ui(r))
        edge_score += 1.3 * float(is_interface_support(r))
        edge_score += 1.2 * edge_seed_conn_n[r]
        edge_score += 0.8 * hop_bonus(edge_hops, r)
        edge_score += 0.6 * area_n[r]
        edge_score += 0.4 * wdeg_n[r]
        edge_score -= 1.9 * float(r in core_anchor_set)
        edge_score -= 1.5 * float(is_chip(r) or is_memory(r) or is_clock(r) or is_power_ic(r))
        edge_score -= 0.8 * float(is_decap(r))

        core_score += 3.0 * float(r in core_anchor_set)
        core_score += 2.0 * float(is_chip(r))
        core_score += 1.8 * float(is_memory(r) or is_clock(r) or is_power_ic(r))
        core_score += 1.0 * float(is_sensor(r))
        core_score += 1.2 * core_seed_conn_n[r]
        core_score += 0.9 * hop_bonus(core_hops, r)
        core_score += 0.8 * chip_neighbor_n[r]
        core_score += 0.7 * wdeg_n[r]
        core_score += 0.5 * area_n[r]
        core_score += 0.6 * float(is_decap(r))
        core_score -= 2.0 * float(r in edge_anchor_set)
        core_score -= 1.3 * float(is_conn(r))
        core_score -= 1.0 * float(is_mech(r))
        core_score -= 0.7 * float(is_interface_support(r))

        balance = 1.0 - min(1.0, abs(edge_score - core_score) / 3.5)
        mid_score += 1.5 * float(is_bridge(r))
        mid_score += 0.8 * float(is_power_ic(r))
        mid_score += 0.7 * float(is_sensor(r))
        mid_score += 0.7 * balance
        mid_score += 0.5 * wdeg_n[r]
        mid_score += 0.4 * degree_n[r]
        mid_score += 0.3 * signal_n[r]
        mid_score -= 0.8 * float(r in edge_anchor_set)
        mid_score -= 0.6 * float(r in core_anchor_set)

        if r in edge_anchor_set:
            lab = "edge"
        elif r in core_anchor_set:
            lab = "core"
        else:
            if edge_score >= core_score + 0.6 and edge_score >= mid_score:
                lab = "edge"
            elif core_score >= edge_score + 0.5 and core_score >= mid_score:
                lab = "core"
            else:
                lab = "mid"
        zone_scores[r] = {"edge": edge_score, "mid": mid_score, "core": core_score}
        zone_label[r] = lab

    remaining: Set[str] = set(refs) - edge_anchor_set - core_anchor_set

    phase0 = list(edge_anchors)
    phase1 = sorted(list(core_anchor_set), key=lambda r: (
        -(core_seed_score[r] - 0.35 * edge_seed_conn_n[r] + 0.10 * area_n[r]),
        r,
    ))

    phase2_pool: Set[str] = set()
    phase3_pool: Set[str] = set()
    phase4_pool: Set[str] = set()
    phase5_pool: Set[str] = set()

    for r in list(remaining):
        core_follow = (
            1.6 * core_seed_conn_n[r]
            + 1.0 * hop_bonus(core_hops, r)
            + 0.9 * support_core_bonus(r)
            + 0.7 * float(zone_label[r] == "core")
            + 0.5 * wdeg_n[r]
            - 0.9 * float(zone_label[r] == "edge")
            - 0.8 * float(r in edge_anchor_set)
        )
        edge_follow = (
            1.6 * edge_seed_conn_n[r]
            + 1.0 * hop_bonus(edge_hops, r)
            + 0.9 * support_edge_bonus(r)
            + 0.7 * float(zone_label[r] == "edge")
            + 0.4 * area_n[r]
            + 0.3 * wdeg_n[r]
            - 0.8 * float(zone_label[r] == "core")
        )
        bridge_score = (
            1.7 * min(core_seed_conn_n[r], edge_seed_conn_n[r])
            + 1.2 * float(is_bridge(r))
            + 0.8 * float(zone_label[r] == "mid")
            + 0.5 * wdeg_n[r]
            + 0.4 * float(is_power_ic(r))
        )
        weak_score = (
            1.0 * float(is_small_passive(r))
            + 0.6 * float((not is_chip(r)) and (not is_conn(r)) and (not is_mech(r)))
            + 0.5 * (1.0 - wdeg_n[r])
            + 0.4 * (1.0 - area_n[r])
        )

        if (is_decap(r) or is_clock(r) or is_memory(r)) and core_follow >= edge_follow - 0.2:
            phase2_pool.add(r)
        elif is_interface_support(r) and edge_follow >= core_follow - 0.1:
            phase3_pool.add(r)
        elif bridge_score >= max(core_follow - 0.15, edge_follow - 0.15):
            phase4_pool.add(r)
        elif core_follow >= edge_follow + 0.2 and core_follow >= bridge_score - 0.1:
            phase2_pool.add(r)
        elif edge_follow >= core_follow + 0.2 and edge_follow >= bridge_score - 0.1:
            phase3_pool.add(r)
        elif zone_label[r] == "core":
            phase2_pool.add(r)
        elif zone_label[r] == "edge":
            phase3_pool.add(r)
        elif weak_score >= 1.7 and bridge_score < 1.0:
            phase5_pool.add(r)
        else:
            phase4_pool.add(r)

    # Keep pools disjoint and exhaustive.
    assigned = phase2_pool | phase3_pool | phase4_pool | phase5_pool
    leftovers = remaining - assigned
    phase4_pool |= leftovers

    def greedy_expand(pool: Set[str], cluster_seed: Set[str], other_seed: Set[str], mode: str) -> List[str]:
        out: List[str] = []
        pool = set(pool)
        cluster = set(cluster_seed)
        placed_local = set(cluster_seed)
        while pool:
            def score(r: str) -> float:
                conn_cluster = conn_to_set_weight(r, cluster)
                conn_cluster_n = conn_cluster / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0))
                conn_seed = conn_to_set_weight(r, cluster_seed) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0))
                conn_other = conn_to_set_weight(r, other_seed) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0)) if other_seed else 0.0
                conn_out = sum(g.neighbor_w.get(r, {}).get(nb, 0.0) for nb in out) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0)) if out else 0.0
                if mode == "core":
                    val = (
                        2.4 * conn_cluster_n
                        + 1.3 * conn_seed
                        + 0.9 * hop_bonus(core_hops, r)
                        + 1.1 * support_core_bonus(r)
                        + 0.7 * float(zone_label[r] == "core")
                        + 0.5 * wdeg_n[r]
                        + 0.3 * conn_out
                        - 0.6 * conn_other
                        - 0.9 * float(r in edge_anchor_set)
                        - 0.5 * float(zone_label[r] == "edge")
                    )
                elif mode == "edge":
                    val = (
                        2.4 * conn_cluster_n
                        + 1.4 * conn_seed
                        + 0.9 * hop_bonus(edge_hops, r)
                        + 1.2 * support_edge_bonus(r)
                        + 0.7 * float(zone_label[r] == "edge")
                        + 0.4 * area_n[r]
                        + 0.3 * wdeg_n[r]
                        + 0.3 * conn_out
                        - 0.6 * conn_other
                        - 0.7 * float(zone_label[r] == "core")
                    )
                else:  # bridge
                    val = (
                        1.6 * min(conn_to_set_weight(r, core_anchor_set), conn_to_set_weight(r, edge_anchor_set)) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0))
                        + 1.1 * float(is_bridge(r))
                        + 0.8 * float(zone_label[r] == "mid")
                        + 0.7 * conn_to_set_weight(r, cluster_seed) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0))
                        + 0.5 * wdeg_n[r]
                        + 0.4 * float(is_power_ic(r))
                        + 0.3 * float(is_sensor(r))
                        - 0.5 * float(is_small_passive(r))
                    )
                return val

            pick = max(sorted(pool), key=lambda r: (score(r), sqrt_area_n[r], degree_n[r], r))
            out.append(pick)
            pool.remove(pick)
            cluster.add(pick)
            placed_local.add(pick)
        return out

    phase2 = greedy_expand(phase2_pool, core_anchor_set, edge_anchor_set, mode="core")
    phase3 = greedy_expand(phase3_pool, edge_anchor_set, core_anchor_set, mode="edge")
    phase4_seed = set(phase2[-min(len(phase2), max(1, bfs_depth)):] + phase3[-min(len(phase3), max(1, bfs_depth)):]) | core_anchor_set | edge_anchor_set
    phase4 = greedy_expand(phase4_pool, phase4_seed, set(), mode="bridge")

    placed_for_tail = set(phase0) | set(phase1) | set(phase2) | set(phase3) | set(phase4)
    tail_candidates = list((phase5_pool | (set(refs) - placed_for_tail)) - set(phase0) - set(phase1) - set(phase2) - set(phase3) - set(phase4))
    tail: List[str] = []
    while tail_candidates:
        def tail_score(r: str) -> Tuple[float, float, float]:
            conn_placed = conn_to_set_weight(r, placed_for_tail | set(tail)) / max(1e-6, max(g.weighted_degree.get(r, 0.0), 1.0))
            val = 1.8 * conn_placed + 0.7 * degree_n[r] + 0.2 * sqrt_area_n[r] - 0.3 * float(is_small_passive(r))
            return (val, degree_n[r], sqrt_area_n[r])
        tail_candidates.sort(key=lambda r: (-tail_score(r)[0], -tail_score(r)[1], -tail_score(r)[2], r))
        pick = tail_candidates.pop(0)
        tail.append(pick)

    # Sequence policy adjusted for anchor / interface stability:
    # 1) reviewed interface / mechanical edge parts
    # 2) core / large anchors
    # 3) power anchors
    # 4) support / bridge / passives
    # 5) misc tail
    reviewed_edge = [r for r in phase0 if review_status(r) in {"approved", "edited", "manual_reviewed", "accepted", "confirmed"}]
    other_edge = [r for r in phase0 if r not in reviewed_edge]
    core_large = [r for r in phase1 if semantic_class(r) in {"core", "interface"} or bool(anchor_ref(r)) or is_bulky(r)]
    power_anchor = [r for r in phase1 if r not in core_large and (semantic_class(r) in {"power", "power_support"} or functional_group(r) == "power_active")]
    core_rest = [r for r in phase1 if r not in core_large and r not in power_anchor]
    seq = reviewed_edge + core_large + power_anchor + core_rest + phase2 + phase4 + other_edge + phase3 + tail

    out: List[str] = []
    seen: Set[str] = set()
    for r in seq:
        if r in comp_by_ref and r not in seen:
            out.append(r)
            seen.add(r)
    for r in refs:
        if r not in seen:
            out.append(r)
            seen.add(r)

    meta = {
        "heuristic_version": "sequence_anchor_interface_priority_v6",
        "bfs_depth": int(bfs_depth),
        "phase_counts": {
            "edge_hard_anchor": len(phase0),
            "core_anchor": len(phase1),
            "core_follower": len(phase2),
            "edge_follower": len(phase3),
            "bridge_mid": len(phase4),
            "weak_small": len(tail),
        },
        "edge_anchors": phase0,
        "core_anchors": phase1,
        "edge_hard_anchors": hard_edge_anchors,
        "edge_soft_anchors": edge_soft_anchors,
        "anchor_overlap_count": len(set(phase0) & set(phase1)),
        "zone_labels": zone_label,
    }
    return (out, meta) if return_meta else out
