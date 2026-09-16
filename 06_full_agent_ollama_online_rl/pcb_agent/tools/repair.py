from __future__ import annotations
from typing import Any
from ..schemas import LayoutDSL, ToolResult

class RepairPlanner:
    name="repair_planner"
    def run(self,drc_report:dict[str,Any],metrics:dict[str,Any],dsl:LayoutDSL)->ToolResult:
        try:
            violations=drc_report.get("violations") or []
            constraint_violations=metrics.get("best_hard_violations") or []
            refs=sorted({r for v in violations for r in (v.get("refs") or [])})
            for v in constraint_violations:
                if isinstance(v,dict):
                    if v.get("ref"):refs.append(str(v["ref"]))
                    refs.extend(str(r) for r in (v.get("refs") or []))
            refs=sorted(set(refs))
            cats={v.get("category","other") for v in violations}
            constraint_types={str(v.get("type","other")) for v in constraint_violations if isinstance(v,dict)}
            patch={};action="local_replacement";reason=[]
            if cats & {"clearance","courtyard","keepout"} or constraint_types & {"clearance","edge","lock"}:
                patch["spacing"]=min(2.0,float(dsl.objective_weights.get("spacing",0.32))*1.25)
                patch["user_preference"]=float(dsl.objective_weights.get("user_preference",2.0))*1.20
                reason.append("geometry or user-constraint violations")
            if "unconnected" in cats:
                patch["hpwl"]=float(dsl.objective_weights.get("hpwl",1.0))*1.15
                patch["weighted_hpwl"]=float(dsl.objective_weights.get("weighted_hpwl",0.2))*1.15
                reason.append("unconnected nets")
            if not refs:
                action="global_replacement";reason.append("no component references in DRC or constraint report")
            plan={
                "action":action,
                "target_refs":refs,
                "reason":", ".join(reason) or "no progress",
                "dsl_patch":{"objective_weights":patch,"movable_refs":refs},
                "freeze_other_refs":bool(refs),
                "repair_window_mm":float(dsl.iteration_policy.repair_window_mm),
            }
            return ToolResult.success(self.name,"Repair plan generated",artifacts={"plan":plan})
        except Exception as exc:return ToolResult.failure(self.name,str(exc))
