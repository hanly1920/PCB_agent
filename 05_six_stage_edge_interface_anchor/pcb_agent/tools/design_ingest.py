from __future__ import annotations
import json, math
from pathlib import Path
from ..schemas import DesignBundle, ToolResult


def _kicad_rot_to_ccw(rot: float) -> float:
    return (-float(rot)) % 360.0


def _component_center_from_fp(fp, bbox) -> tuple[float, float, float]:
    r = _kicad_rot_to_ccw(float(getattr(fp, "rot", 0.0)))
    x = float(getattr(fp, "at", (0.0, 0.0))[0])
    y = float(getattr(fp, "at", (0.0, 0.0))[1])
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cx = 0.5 * (float(x0) + float(x1))
        cy = 0.5 * (float(y0) + float(y1))
        th = math.radians(r % 360.0)
        c = math.cos(th)
        ss = math.sin(th)
        x += cx * c - cy * ss
        y += cx * ss + cy * c
    return float(x), float(y), float(r)


class DesignIngestTool:
    name = "design_ingest"

    def run(self, bundle_dir: str | Path, output_dir: str | Path) -> ToolResult:
        try:
            bundle = DesignBundle.load(bundle_dir)
            if not bundle.board_path.exists():
                raise FileNotFoundError(f"Board not found: {bundle.board_path}")
            out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
            task_path = bundle.task_json_path
            if task_path is None or not task_path.exists():
                task_path = out / "task.generated.json"
                self._board_to_task(bundle.board_path, task_path)
            return ToolResult.success(self.name, "Design bundle validated", artifacts={"board_path": str(bundle.board_path), "task_json_path": str(task_path), "bundle": bundle.model_dump(mode="json")}, observations=["Existing structured task reused" if bundle.task_json_path and bundle.task_json_path.exists() else "Structured task generated from KiCad board"])
        except Exception as exc:
            return ToolResult.failure(self.name, str(exc))

    @staticmethod
    def _board_to_task(board_path: Path, output_path: Path) -> None:
        from pcbplace.kicad_parser import parse_kicad_pcb
        board = parse_kicad_pcb(board_path.read_text(encoding="utf-8", errors="replace"))
        components=[]; nets: dict[str,list[str]]={}
        for fp in board.footprints:
            if not fp.ref:
                continue
            bb = fp.bbox_local or (-0.5,-0.5,0.5,0.5)
            size=[max(0.1, float(bb[2])-float(bb[0])), max(0.1, float(bb[3])-float(bb[1]))]
            cx = 0.5 * (float(bb[0]) + float(bb[2]))
            cy = 0.5 * (float(bb[1]) + float(bb[3]))
            pads=[]
            for pad in (fp.pads or []):
                pads.append({"name": pad.name, "net": pad.net, "rel_mm": [float(pad.at[0]-cx), float(pad.at[1]-cy)]})
                if pad.net:
                    nets.setdefault(pad.net, []).append(f"{fp.ref}.{pad.name}")
            typ = "interface" if fp.ref.upper().startswith(("J","P")) else "misc"
            locked = bool(getattr(fp, "locked", False))
            comp={"ref":fp.ref,"footprint":fp.footprint,"type":typ,"size_mm":size,"pads":pads,"allowed_sides":[],"fixed":bool(locked),"semantic_class":"interface" if typ=="interface" else "other","region_type":"edge" if typ=="interface" else "free","functional_group":"interface" if typ=="interface" else "other","side_preference":"free","critical_nets":[],"critical_neighbors":[],"anchor_ref":None,"subzone":"free","same_side_group":None,"boundary_order":None,"placement_role":"member"}
            if locked:
                fx, fy, frot = _component_center_from_fp(fp, bb)
                comp.update({"locked": True, "fixed_xy_mm": [fx, fy], "fixed_rot": frot, "fixed_source": "kicad_locked"})
            components.append(comp)
        task={"board":{"bbox_mm":list(map(float,board.bbox)),"grid_mm":1.0},"components":components,"nets":nets,"graph":{"sequence":[c["ref"] for c in components],"sequence_policy":"stored","sequence_source":"generated_from_kicad"},"meta":{"generated_by":"pcb_agent.DesignIngestTool","source_board":str(board_path)}}
        output_path.write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding="utf-8")
