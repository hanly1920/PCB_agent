from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Token = str

def _tokenize(s: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == '(' or c == ')':
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n:
                if s[j] == '"' and s[j-1] != '\\':
                    break
                buf.append(s[j])
                j += 1
            tokens.append('"' + ''.join(buf) + '"')
            i = j + 1
            continue
        j = i
        while j < n and (not s[j].isspace()) and s[j] not in '()':
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens

def _parse(tokens: List[Token]) -> Any:
    stack: List[List[Any]] = []
    cur: List[Any] = []
    for t in tokens:
        if t == '(':
            stack.append(cur)
            cur = []
        elif t == ')':
            if not stack:
                raise ValueError('unbalanced )')
            completed = cur
            cur = stack.pop()
            cur.append(completed)
        else:
            cur.append(t)
    if stack:
        raise ValueError('unbalanced (')
    if len(cur) != 1:
        return cur
    return cur[0]

def _unquote(tok: str) -> str:
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    return tok

def _as_float(tok: str, default: float = 0.0) -> float:
    try:
        return float(tok)
    except Exception:
        return default

def _bbox_union(b1: Optional[Tuple[float,float,float,float]], b2: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    if b1 is None:
        return b2
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

def _bbox_from_points(pts: List[Tuple[float,float]]) -> Optional[Tuple[float,float,float,float]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

def _circle_from_3pts(p1, p2, p3):
    # return center (cx,cy) and radius r; None if collinear
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1**2 - x2**2) + (y1**2 - y2**2)) / 2.0
    f = ((x1**2 - x3**2) + (y1**2 - y3**2)) / 2.0
    det = a*d - b*c
    if abs(det) < 1e-9:
        return None
    cx = (d*e - b*f) / det
    cy = (-c*e + a*f) / det
    r = math.hypot(cx-x1, cy-y1)
    return (cx,cy,r)

def _angle(x, y, cx, cy):
    return math.atan2(y-cy, x-cx)

def _is_angle_between(a, start, end, ccw: bool) -> bool:
    # Normalize into [0, 2pi)
    tau = 2*math.pi
    a = a % tau
    start = start % tau
    end = end % tau
    if ccw:
        if start <= end:
            return start <= a <= end
        else:
            return a >= start or a <= end
    else:
        # clockwise: between when NOT between in ccw from start to end
        if start <= end:
            return not (start <= a <= end)
        else:
            return not (a >= start or a <= end)

def _arc_bbox(start, mid, end) -> Optional[Tuple[float,float,float,float]]:
    circ = _circle_from_3pts(start, mid, end)
    if circ is None:
        return _bbox_from_points([start, mid, end])
    cx,cy,r = circ
    ang_s = _angle(start[0], start[1], cx, cy)
    ang_m = _angle(mid[0], mid[1], cx, cy)
    ang_e = _angle(end[0], end[1], cx, cy)

    # determine direction: if mid is on ccw path from start to end, then ccw
    ccw = _is_angle_between(ang_m, ang_s, ang_e, ccw=True)
    # candidate angles at extrema
    candidates = [ang_s, ang_e]
    for k in [0, math.pi/2, math.pi, 3*math.pi/2]:
        if _is_angle_between(k, ang_s, ang_e, ccw=ccw):
            candidates.append(k)
    pts = [(cx + r*math.cos(a), cy + r*math.sin(a)) for a in candidates]
    return _bbox_from_points(pts)

@dataclass
class Pad:
    name: str
    at: Tuple[float,float]
    rot: float
    size: Tuple[float,float]
    net: str

@dataclass
class Footprint:
    ref: str
    footprint: str
    at: Tuple[float,float]
    rot: float  # raw KiCad value (clockwise in .kicad_pcb); convert as needed in exporters
    pads: List[Pad]
    # Chosen local bbox in footprint coordinates (mm):
    #   - Prefer Courtyard (F.CrtYd/B.CrtYd) if present
    #   - Fallback to pad bbox
    bbox_local: Optional[Tuple[float,float,float,float]]
    # Optional: keep both sources for downstream decision-making
    bbox_courtyard: Optional[Tuple[float,float,float,float]] = None
    bbox_pads: Optional[Tuple[float,float,float,float]] = None
@dataclass
class Board:
    bbox: Tuple[float,float,float,float]
    footprints: List[Footprint]
    nets: Dict[str, int]

def parse_kicad_pcb(text: str) -> Board:
    """Best-effort KiCad v6/v7 parser.
    Improvements vs baseline:
      - Edge.Cuts bbox supports gr_line/gr_rect/gr_poly/gr_circle/gr_arc.
      - Footprint local bbox derived from F.CrtYd/B.CrtYd if present, else from pad bboxes.
      - Pad size and pad local rotation are parsed.
    """
    ast = _parse(_tokenize(text))

    nets: Dict[str,int] = {}
    def walk_nets(node: Any):
        if isinstance(node, list):
            if node and node[0] == 'net' and len(node) >= 3:
                net_id = int(node[1]) if re.match(r'^-?\d+$', str(node[1])) else 0
                net_name = _unquote(str(node[2]))
                nets[net_name] = net_id
            for ch in node:
                walk_nets(ch)
    walk_nets(ast)

    # Edge.Cuts bbox
    bbox: Optional[Tuple[float,float,float,float]] = None

    def add_points(pts: List[Tuple[float,float]]):
        nonlocal bbox
        b = _bbox_from_points(pts)
        if b is None:
            return
        bbox = _bbox_union(bbox, b)

    def walk_edge(node: Any):
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        layer = None

        def get_layer(n: List[Any]) -> Optional[str]:
            for it in n[1:]:
                if isinstance(it, list) and it and it[0] == 'layer' and len(it) >= 2:
                    return _unquote(str(it[1]))
            return None

        if head in ('gr_line','gr_rect','gr_poly','gr_circle','gr_arc'):
            layer = get_layer(node)
            if layer == 'Edge.Cuts':
                if head in ('gr_line','gr_rect'):
                    pts = []
                    for it in node[1:]:
                        if isinstance(it, list) and it:
                            if it[0] == 'start' and len(it) >= 3:
                                pts.append((_as_float(it[1]), _as_float(it[2])))
                            if it[0] == 'end' and len(it) >= 3:
                                pts.append((_as_float(it[1]), _as_float(it[2])))
                    add_points(pts)
                elif head == 'gr_poly':
                    pts = []
                    for it in node[1:]:
                        if isinstance(it, list) and it and it[0] == 'pts':
                            for p in it[1:]:
                                if isinstance(p, list) and p and p[0] == 'xy' and len(p) >= 3:
                                    pts.append((_as_float(p[1]), _as_float(p[2])))
                    add_points(pts)
                elif head == 'gr_circle':
                    center = None
                    end = None
                    for it in node[1:]:
                        if isinstance(it, list) and it:
                            if it[0] == 'center' and len(it) >= 3:
                                center = (_as_float(it[1]), _as_float(it[2]))
                            if it[0] == 'end' and len(it) >= 3:
                                end = (_as_float(it[1]), _as_float(it[2]))
                    if center and end:
                        r = math.hypot(end[0]-center[0], end[1]-center[1])
                        add_points([(center[0]-r, center[1]-r), (center[0]+r, center[1]+r)])
                elif head == 'gr_arc':
                    start = mid = end = None
                    for it in node[1:]:
                        if isinstance(it, list) and it:
                            if it[0] == 'start' and len(it) >= 3:
                                start = (_as_float(it[1]), _as_float(it[2]))
                            if it[0] == 'mid' and len(it) >= 3:
                                mid = (_as_float(it[1]), _as_float(it[2]))
                            if it[0] == 'end' and len(it) >= 3:
                                end = (_as_float(it[1]), _as_float(it[2]))
                    if start and mid and end:
                        b = _arc_bbox(start, mid, end)
                        if b:
                            nonlocal bbox
                            bbox = _bbox_union(bbox, b)
        for ch in node[1:]:
            walk_edge(ch)

    walk_edge(ast)
    if bbox is None:
        bbox = (0.0, 0.0, 100.0, 100.0)

    footprints: List[Footprint] = []

    def walk_fp(node: Any):
        if not isinstance(node, list) or not node:
            return
        # KiCad 5 used "module" for footprints; KiCad 6+ uses "footprint".
        if node[0] in ('footprint', 'module'):
            fp_name = _unquote(str(node[1])) if len(node) > 1 else ''
            ref = ''
            at = (0.0, 0.0); rot = 0.0
            pads: List[Pad] = []
            # local geometry points for courtyard bbox
            crtyd_pts: List[Tuple[float,float]] = []
            pad_bbox: Optional[Tuple[float,float,float,float]] = None

            def layer_of(n: List[Any]) -> Optional[str]:
                for it in n[1:]:
                    if isinstance(it, list) and it and it[0] == 'layer' and len(it) >= 2:
                        return _unquote(str(it[1]))
                return None

            def add_local_points(pts: List[Tuple[float,float]]):
                crtyd_pts.extend(pts)

            for item in node[2:]:
                if isinstance(item, list) and item:
                    if item[0] == 'at' and len(item) >= 3:
                        at = (_as_float(item[1]), _as_float(item[2]))
                        if len(item) >= 4:
                            rot = _as_float(item[3])
                    if item[0] == 'property' and len(item) >= 3:
                        if _unquote(str(item[1])) in ('Reference','REF','Ref'):
                            ref = _unquote(str(item[2]))
                    if item[0] == 'fp_text' and len(item) >= 3:
                        if str(item[1]) == 'reference':
                            ref = _unquote(str(item[2]))

                    # courtyard primitives inside footprint
                    if item[0] in ('fp_line','fp_rect','fp_poly','fp_circle','fp_arc'):
                        lyr = layer_of(item)
                        if lyr in ('F.CrtYd','B.CrtYd'):
                            if item[0] in ('fp_line','fp_rect'):
                                pts = []
                                for it in item[1:]:
                                    if isinstance(it, list) and it:
                                        if it[0] == 'start' and len(it) >= 3:
                                            pts.append((_as_float(it[1]), _as_float(it[2])))
                                        if it[0] == 'end' and len(it) >= 3:
                                            pts.append((_as_float(it[1]), _as_float(it[2])))
                                add_local_points(pts)
                            elif item[0] == 'fp_poly':
                                pts = []
                                for it in item[1:]:
                                    if isinstance(it, list) and it and it[0] == 'pts':
                                        for p in it[1:]:
                                            if isinstance(p, list) and p and p[0] == 'xy' and len(p) >= 3:
                                                pts.append((_as_float(p[1]), _as_float(p[2])))
                                add_local_points(pts)
                            elif item[0] == 'fp_circle':
                                center = None
                                endp = None
                                for it in item[1:]:
                                    if isinstance(it, list) and it:
                                        if it[0] == 'center' and len(it) >= 3:
                                            center = (_as_float(it[1]), _as_float(it[2]))
                                        if it[0] == 'end' and len(it) >= 3:
                                            endp = (_as_float(it[1]), _as_float(it[2]))
                                if center and endp:
                                    r = math.hypot(endp[0]-center[0], endp[1]-center[1])
                                    add_local_points([(center[0]-r, center[1]-r), (center[0]+r, center[1]+r)])
                            elif item[0] == 'fp_arc':
                                start = mid = endp = None
                                for it in item[1:]:
                                    if isinstance(it, list) and it:
                                        if it[0] == 'start' and len(it) >= 3:
                                            start = (_as_float(it[1]), _as_float(it[2]))
                                        if it[0] == 'mid' and len(it) >= 3:
                                            mid = (_as_float(it[1]), _as_float(it[2]))
                                        if it[0] == 'end' and len(it) >= 3:
                                            endp = (_as_float(it[1]), _as_float(it[2]))
                                if start and mid and endp:
                                    b = _arc_bbox(start, mid, endp)
                                    if b:
                                        add_local_points([(b[0], b[1]), (b[2], b[3])])

                    if item[0] == 'pad' and len(item) >= 2:
                        pad_name = _unquote(str(item[1]))
                        pad_at = (0.0, 0.0)
                        pad_rot = 0.0
                        pad_size = (0.0, 0.0)
                        net_name = ''
                        for pit in item[2:]:
                            if isinstance(pit, list) and pit:
                                if pit[0] == 'at' and len(pit) >= 3:
                                    pad_at = (_as_float(pit[1]), _as_float(pit[2]))
                                    if len(pit) >= 4:
                                        pad_rot = _as_float(pit[3])
                                if pit[0] == 'size' and len(pit) >= 3:
                                    pad_size = (_as_float(pit[1]), _as_float(pit[2]))
                                if pit[0] == 'net' and len(pit) >= 3:
                                    net_name = _unquote(str(pit[2]))
                        pads.append(Pad(name=pad_name, at=pad_at, rot=pad_rot, size=pad_size, net=net_name))

                        # pad bbox (local): include pad rotation
                        sx, sy = pad_size
                        if sx > 0 and sy > 0:
                            ang = math.radians((-pad_rot) % 360.0)  # KiCad angles are clockwise in file coords; convert to CCW
                            ca, sa = math.cos(ang), math.sin(ang)
                            # corners of axis-aligned pad before rotation
                            corners = [(-sx/2,-sy/2), (-sx/2,sy/2), (sx/2,-sy/2), (sx/2,sy/2)]
                            pts = []
                            for x0,y0 in corners:
                                xr = ca*x0 - sa*y0 + pad_at[0]
                                yr = sa*x0 + ca*y0 + pad_at[1]
                                pts.append((xr, yr))
                            b = _bbox_from_points(pts)
                            if b:
                                pad_bbox = _bbox_union(pad_bbox, b)

            crtyd_bbox = _bbox_from_points(crtyd_pts)
            local_bbox = crtyd_bbox if crtyd_bbox is not None else pad_bbox
            if ref:
                footprints.append(Footprint(ref=ref, footprint=fp_name, at=at, rot=rot, pads=pads, bbox_local=local_bbox, bbox_courtyard=crtyd_bbox, bbox_pads=pad_bbox))
        for ch in node[1:]:
            walk_fp(ch)

    walk_fp(ast)

    return Board(bbox=bbox, footprints=footprints, nets=nets)
