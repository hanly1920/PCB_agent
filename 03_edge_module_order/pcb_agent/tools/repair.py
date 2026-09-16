from __future__ import annotations
from typing import Any
from ..schemas import LayoutDSL, ToolResult

class RepairPlanner:
    name="repair_planner"
    def run(self,drc_report:dict[str,Any],metrics:dict[str,Any],dsl:LayoutDSL)->ToolResult:
        try:
            violations=drc_report.get("violations") or [];refs=sorted({r for v in violations for r in (v.get("refs") or [])});cats={v.get("category","other") for v in violations};patch={};action="local_replacement";reason=[]
            if cats & {"clearance","courtyard","keepout"}:patch["spacing"]=min(2.0,float(dsl.objective_weights.get("spacing",0.32))*1.25);reason.append("geometry violations")
            if "unconnected" in cats:patch["hpwl"]=float(dsl.objective_weights.get("hpwl",1.0))*1.15;reason.append("unconnected nets")
            if not refs:action="global_replacement";reason.append("no component references in DRC")
            plan={"action":action,"target_refs":refs,"reason":", ".join(reason) or "no progress","dsl_patch":{"objective_weights":patch},"freeze_other_refs":bool(refs)}
            return ToolResult.success(self.name,"Repair plan generated",artifacts={"plan":plan})
        except Exception as exc:return ToolResult.failure(self.name,str(exc))
