from __future__ import annotations
import fnmatch, json, math
from pathlib import Path
from typing import Any
from ..schemas import LayoutCandidate, LayoutDSL, ToolResult

class LayoutAnalyzer:
    name="layout_analyzer"
    def run(self,candidates:list[dict[str,Any]]|list[LayoutCandidate],task_json_path:str|Path,dsl:LayoutDSL)->ToolResult:
        try:
            cs=[c if isinstance(c,LayoutCandidate) else LayoutCandidate.model_validate(c) for c in candidates]
            task=json.loads(Path(task_json_path).read_text(encoding="utf-8")); bbox=list(map(float,task["board"]["bbox_mm"])); scores=[]
            for c in cs:
                pref=self._preference_score(c.placements,bbox,dsl)
                illegal=0 if c.legal else 1
                total=(0 if math.isfinite(c.objective) else 1e12)+float(c.objective if math.isfinite(c.objective) else 1e12)+illegal*1e9-pref*float(dsl.objective_weights.get("user_preference",2.0))
                scores.append({"candidate_id":c.candidate_id,"score":total,"preference_score":pref,"legal":c.legal,"complete":c.complete,"objective":c.objective})
            scores.sort(key=lambda x:(not x["complete"],not x["legal"],x["score"]))
            best=next(c for c in cs if c.candidate_id==scores[0]["candidate_id"])
            return ToolResult.success(self.name,f"Selected {best.candidate_id}",artifacts={"best_candidate":best.model_dump(mode="json"),"ranking":scores},metrics={"best_score":scores[0]["score"],"candidate_count":len(cs)})
        except Exception as exc:return ToolResult.failure(self.name,str(exc))

    @staticmethod
    def _preference_score(p:dict[str,dict[str,Any]],bbox:list[float],dsl:LayoutDSL)->float:
        x0,y0,x1,y1=bbox; cx=(x0+x1)/2; cy=(y0+y1)/2; diag=max(1e-6,math.hypot(x1-x0,y1-y0)); score=0.0
        for c in dsl.soft_constraints+dsl.region_constraints:
            refs=[r for pat in c.refs for r in p if fnmatch.fnmatchcase(r,pat)]
            if c.type=="prefer_region" and c.region=="center":
                for r in refs: score+=max(0.0,1.0-math.hypot(float(p[r]["x"])-cx,float(p[r]["y"])-cy)/diag)*(c.weight or 1.0)
            elif c.type=="near" and c.target in p:
                tx,ty=float(p[c.target]["x"]),float(p[c.target]["y"]); rad=max(1e-6,float(c.radius_mm or 6.0))
                for r in refs: score+=max(0.0,1.0-math.hypot(float(p[r]["x"])-tx,float(p[r]["y"])-ty)/rad)*(c.weight or 1.0)
        return score
