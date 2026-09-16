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
                report=self._constraint_report(c.placements,task,bbox,dsl)
                pref=float(report["soft_score"])
                hard_count=len(report["hard_violations"])
                illegal=0 if c.legal else 1
                objective=float(c.objective if math.isfinite(c.objective) else 1e12)
                total=objective+illegal*1e9+hard_count*1e8-pref*float(dsl.objective_weights.get("user_preference",2.0))
                scores.append({"candidate_id":c.candidate_id,"score":total,"soft_score":pref,"hard_violation_count":hard_count,"hard_violations":report["hard_violations"],"legal":c.legal,"complete":c.complete,"objective":c.objective})
            scores.sort(key=lambda x:(not x["complete"],not x["legal"],x["hard_violation_count"],x["score"]))
            best=next(c for c in cs if c.candidate_id==scores[0]["candidate_id"])
            return ToolResult.success(self.name,f"Selected {best.candidate_id}",artifacts={"best_candidate":best.model_dump(mode="json"),"ranking":scores},metrics={"best_score":scores[0]["score"],"best_soft_score":scores[0]["soft_score"],"best_hard_violation_count":scores[0]["hard_violation_count"],"candidate_count":len(cs)})
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

    @classmethod
    def _constraint_report(cls,p:dict[str,dict[str,Any]],task:dict[str,Any],bbox:list[float],dsl:LayoutDSL)->dict[str,Any]:
        comps={str(c.get("ref")):c for c in (task.get("components") or []) if c.get("ref")}
        soft=0.0
        violations:list[dict[str,Any]]=[]

        for ref,comp in comps.items():
            if not bool(comp.get("fixed")) or ref not in p:
                continue
            xy=comp.get("fixed_xy_mm")
            if isinstance(xy,(list,tuple)) and len(xy)>=2:
                dist=math.hypot(float(p[ref]["x"])-float(xy[0]),float(p[ref]["y"])-float(xy[1]))
                if dist>0.25:
                    violations.append({"type":"lock","ref":ref,"distance_mm":dist})

        for c in dsl.hard_constraints+dsl.soft_constraints:
            ctype=str(c.type or "").lower()
            refs=[r for pat in c.refs for r in p if fnmatch.fnmatchcase(r,pat)]
            if ctype=="edge":
                side=str(c.side or "").lower()
                for ref in refs:
                    edge_score=cls._edge_score(float(p[ref]["x"]),float(p[ref]["y"]),bbox,side)
                    soft+=edge_score*float(c.weight or 1.0)
                    if c.priority=="hard" and edge_score<0.35:
                        violations.append({"type":"edge","ref":ref,"side":side,"score":edge_score})
            elif ctype=="prefer_region":
                region=str(c.region or "").lower()
                for ref in refs:
                    score=cls._region_score(float(p[ref]["x"]),float(p[ref]["y"]),bbox,region)
                    soft+=score*float(c.weight or 1.0)
            elif ctype=="near" and c.target in p:
                tx,ty=float(p[c.target]["x"]),float(p[c.target]["y"]); rad=max(1e-6,float(c.radius_mm or 6.0))
                for ref in refs:
                    dist=math.hypot(float(p[ref]["x"])-tx,float(p[ref]["y"])-ty)
                    near_score=max(0.0,1.0-dist/rad)
                    soft+=near_score*float(c.weight or 1.0)
            elif ctype=="clearance" and c.value_mm is not None:
                clearance=float(c.value_mm)
                refs_to_check=sorted(p)
                for i,ra in enumerate(refs_to_check):
                    for rb in refs_to_check[i+1:]:
                        ca=comps.get(ra) or {}; cb=comps.get(rb) or {}
                        if cls._component_gap_mm(p[ra],ca,p[rb],cb)<clearance:
                            violations.append({"type":"clearance","refs":[ra,rb],"required_mm":clearance})
        return {"soft_score":soft,"hard_violations":violations}

    @staticmethod
    def _edge_score(x:float,y:float,bbox:list[float],side:str)->float:
        x0,y0,x1,y1=bbox; width=max(1e-6,x1-x0); height=max(1e-6,y1-y0)
        band=max(2.0,0.20*(width if side in {"left","right"} else height))
        if side=="left": dist=x-x0
        elif side=="right": dist=x1-x
        elif side=="top": dist=y-y0
        elif side=="bottom": dist=y1-y
        else: return 0.0
        return max(0.0,min(1.0,1.0-dist/max(1e-6,band)))

    @staticmethod
    def _region_score(x:float,y:float,bbox:list[float],region:str)->float:
        x0,y0,x1,y1=bbox; cx=(x0+x1)/2; cy=(y0+y1)/2; diag=max(1e-6,math.hypot(x1-x0,y1-y0))
        if region in {"center","core"}:
            return max(0.0,1.0-math.hypot(x-cx,y-cy)/(0.35*diag))
        return LayoutAnalyzer._edge_score(x,y,bbox,region)

    @staticmethod
    def _component_gap_mm(pa:dict[str,Any],ca:dict[str,Any],pb:dict[str,Any],cb:dict[str,Any])->float:
        ax,ay=float(pa["x"]),float(pa["y"]); bx,by=float(pb["x"]),float(pb["y"])
        aw,ah=(ca.get("size_mm") or [0.0,0.0])[:2]; bw,bh=(cb.get("size_mm") or [0.0,0.0])[:2]
        dx=abs(ax-bx)-0.5*float(aw)-0.5*float(bw)
        dy=abs(ay-by)-0.5*float(ah)-0.5*float(bh)
        return math.hypot(max(0.0,dx),max(0.0,dy)) if dx>0 or dy>0 else min(dx,dy)
