from pathlib import Path
from pcb_agent.tools.kicad_bridge import KicadBridge
from pcbplace.kicad_parser import parse_kicad_pcb

def test_apply_placement_roundtrip(tmp_path):
    src=Path(__file__).parent/"fixtures"/"tiny.kicad_pcb";dst=tmp_path/"placed.kicad_pcb"
    result=KicadBridge().apply_placement(src,{"U1":{"x":12.5,"y":13.5,"rotation":90,"side":"F.Cu"}},dst)
    assert result.ok
    board=parse_kicad_pcb(dst.read_text())
    u1=next(fp for fp in board.footprints if fp.ref=="U1")
    assert u1.at==(12.5,13.5);assert u1.rot==90
