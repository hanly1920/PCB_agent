from __future__ import annotations
import fnmatch, json, math
from pathlib import Path
from typing import Any
from ..config import PlacementConfig
from ..schemas import LayoutCandidate, LayoutDSL, ToolResult


def _xy_from_value(value: Any) -> tuple[float, float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for keys in (("x", "y"), ("cx", "cy")):
            if all(k in value for k in keys):
                return (float(value[keys[0]]), float(value[keys[1]]))
        for key in ("xy_mm", "center_mm", "position_mm", "pos_mm", "at_mm", "fixed_xy_mm"):
            got = _xy_from_value(value.get(key))
            if got is not None:
                return got
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def _rot_from_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for key in ("rot", "rotation", "rot_deg", "angle", "angle_deg", "fixed_rot"):
            if value.get(key) not in (None, ""):
                return float(value.get(key)) % 360.0
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[2]) % 360.0
    try:
        return float(value) % 360.0
    except Exception:
        return None


def _component_existing_xy(comp: dict[str, Any]) -> tuple[float, float] | None:
    for value in (
        comp.get("fixed_xy_mm"), comp.get("fixed_position_mm"), comp.get("locked_xy_mm"),
        comp.get("fixed"), comp.get("placement"), comp.get("layout"), comp.get("expert"),
    ):
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    return None


def _component_existing_rot(comp: dict[str, Any]) -> float:
    for value in (
        comp.get("fixed_rot"), comp.get("locked_rot"), comp.get("fixed"), comp.get("placement"), comp.get("layout"), comp.get("expert"),
    ):
        rot = _rot_from_value(value)
        if rot is not None:
            return float(rot)
    return 0.0


class PlacementModelTool:
    name = "placement_model"
    def __init__(self, config: PlacementConfig): self.config=config

    def validate_checkpoint(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(checkpoint_path or self.config.checkpoint_path or "")
        if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
        import torch
        ckpt=torch.load(path,map_location="cpu",weights_only=False)
        if not isinstance(ckpt,dict) or "model_state" not in ckpt: raise ValueError("Checkpoint must be a dict containing model_state")
        return {"path":str(path),"format_version":ckpt.get("format_version"),"obs_dim":ckpt.get("obs_dim"),"action_feat_dim":ckpt.get("action_feat_dim"),"model_cfg":ckpt.get("model_cfg"),"max_tokens":ckpt.get("max_tokens"),"sequence_policy":ckpt.get("sequence_policy"),"action_scoring":ckpt.get("action_scoring"),"env_config":ckpt.get("env_config"),"parameter_tensors":len(ckpt.get("model_state") or {})}

    @staticmethod
    def _dsl_locked_patterns(dsl: LayoutDSL) -> list[str]:
        refs = [str(r) for r in (dsl.locked_refs or []) if str(r).strip()]
        for c in list(dsl.hard_constraints or []) + list(dsl.soft_constraints or []):
            if str(c.type).lower() in {"lock", "locked", "fix", "fixed", "freeze", "frozen"}:
                refs.extend(str(r) for r in (c.refs or []) if str(r).strip())
        return sorted(set(refs))

    @staticmethod
    def _matches(ref: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(ref, pat) for pat in patterns)

    def _task_with_dsl_locks(self, task_json_path: Path, dsl: LayoutDSL) -> Path:
        patterns = self._dsl_locked_patterns(dsl)
        if not patterns:
            return task_json_path
        data = json.loads(task_json_path.read_text(encoding="utf-8"))
        missing: list[str] = []
        matched: list[str] = []
        for comp in data.get("components", []):
            ref = str(comp.get("ref") or "")
            if not ref or not self._matches(ref, patterns):
                continue
            matched.append(ref)
            xy = _component_existing_xy(comp)
            if xy is None:
                missing.append(ref)
                continue
            comp["fixed"] = True
            comp["locked"] = True
            comp["fixed_xy_mm"] = [float(xy[0]), float(xy[1])]
            comp["fixed_rot"] = float(_component_existing_rot(comp))
            comp["fixed_source"] = "dsl_locked_ref"
        if not matched:
            raise ValueError(f"DSL locked_refs did not match any components: {patterns}")
        if missing:
            raise ValueError(
                "DSL locked_refs require existing coordinates. Missing fixed_xy_mm/placement/layout/expert.xy_mm for: "
                + ", ".join(sorted(missing))
            )
        out = task_json_path.with_name(task_json_path.stem + ".agent_fixed" + task_json_path.suffix)
        data.setdefault("meta", {})["dsl_locked_refs"] = patterns
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def run(self, task_json_path: str | Path, dsl: LayoutDSL, *, candidate_count: int | None = None) -> ToolResult:
        try:
            count=max(1,int(candidate_count or dsl.iteration_policy.candidate_count))
            task_path = self._task_with_dsl_locks(Path(task_json_path), dsl)
            if self.config.mock:
                candidates=self._mock_candidates(task_path, count)
                return ToolResult.success(self.name,"Generated deterministic mock candidates",artifacts={"candidates":[c.model_dump(mode="json") for c in candidates],"task_json_path":str(task_path)},metrics={"candidate_count":len(candidates)},observations=["Mock placement is intended for orchestration tests, not placement quality evaluation"])
            meta=self.validate_checkpoint()
            candidates=[]
            from pcbplace.infer import infer_layout
            presets=[self.config.layout_preset,"balanced","hpwl","neat","edge_strict"]
            for i in range(count):
                preset=presets[i%len(presets)]
                beam=max(1,self.config.beam_width + (i%2))
                result=infer_layout(str(task_path),str(self.config.checkpoint_path),device=self.config.device,beam_width=beam,beam_topk=self.config.beam_topk,return_metrics=True,strict_state_dict=self.config.strict_state_dict,postprocess=True,layout_preset=preset,objective_overrides=dsl.objective_overrides(),sequence_policy="checkpoint")
                candidates.append(self._normalize(result,i,source=f"pcbplace:{preset}:beam{beam}"))
            return ToolResult.success(self.name,"Placement inference completed",artifacts={"candidates":[c.model_dump(mode="json") for c in candidates],"checkpoint_meta":meta,"task_json_path":str(task_path)},metrics={"candidate_count":len(candidates),"complete_count":sum(c.complete for c in candidates)})
        except Exception as exc:
            return ToolResult.failure(self.name,str(exc))

    @staticmethod
    def _normalize(result: dict[str,Any], index:int, source:str) -> LayoutCandidate:
        placements={}
        for ref,value in (result.get("placed") or {}).items():
            x,y,rot=value
            placements[str(ref)]={"x":float(x),"y":float(y),"rotation":float(rot),"side":"F.Cu"}
        complete=bool(result.get("complete")); legal=complete and float((result.get("metrics") or {}).get("illegal_count",0) or 0)==0
        return LayoutCandidate(candidate_id=f"candidate-{index:03d}",placements=placements,objective=float(result.get("objective",1e12)),metrics=dict(result.get("metrics") or {}),legal=legal,complete=complete,source=source,raw=result)

    @staticmethod
    def _mock_candidates(task_path: Path, count:int) -> list[LayoutCandidate]:
        task=json.loads(task_path.read_text(encoding="utf-8")); bbox=task["board"]["bbox_mm"]; comps=task.get("components",[])
        x0,y0,x1,y1=map(float,bbox); margin=2.0; cols=max(1,int(math.ceil(math.sqrt(max(1,len(comps)))))); spacing_x=max(1.0,(x1-x0-2*margin)/max(1,cols)); rows=max(1,int(math.ceil(len(comps)/cols))); spacing_y=max(1.0,(y1-y0-2*margin)/max(1,rows))
        out=[]
        for k in range(count):
            placements={}
            for i,c in enumerate(comps):
                ref=str(c["ref"])
                xy = _component_existing_xy(c) if bool(c.get("fixed")) else None
                if xy is not None:
                    placements[ref]={"x":float(xy[0]),"y":float(xy[1]),"rotation":float(_component_existing_rot(c)),"side":"F.Cu"}
                    continue
                col=i%cols; row=i//cols; jitter=(k*0.15)
                placements[ref]={"x":min(x1-margin,x0+margin+(col+0.5)*spacing_x+jitter),"y":min(y1-margin,y0+margin+(row+0.5)*spacing_y+jitter),"rotation":float((k%4)*90),"side":"F.Cu"}
            out.append(LayoutCandidate(candidate_id=f"mock-{k:03d}",placements=placements,objective=float(k),metrics={"mock":True,"component_count":len(placements)},legal=True,complete=True,source="mock_grid"))
        return out
