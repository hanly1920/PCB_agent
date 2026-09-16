from __future__ import annotations
import json,re
from pathlib import Path
from typing import Any
from ..config import KiCadConfig
from ..schemas import DRCReport, DRCViolation, ToolResult
from .command import CommandRunner

class KicadDRCTool:
    name="kicad_drc"
    def __init__(self,config:KiCadConfig):self.config=config;self.runner=CommandRunner()
    def run(self,board_path:str|Path,output_dir:str|Path)->ToolResult:
        try:
            out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);report_path=out/"drc.json"
            if self.config.mock_drc:
                report=DRCReport(drc_clean=True,error_count=0,warning_count=0,raw={"mock":True});report_path.write_text(json.dumps(report.model_dump(mode="json"),indent=2),encoding="utf-8")
                return ToolResult.success(self.name,"Mock DRC clean",artifacts={"drc_path":str(report_path),"report":report.model_dump(mode="json")},metrics={"drc_clean":True,"error_count":0,"warning_count":0})
            cmd=[self.config.cli,"pcb","drc","--format","json","--exit-code-violations","--output",str(report_path),str(board_path)]
            res=self.runner.run(self.name,cmd,timeout_sec=self.config.drc_timeout_sec,allowed_returncodes={0,5})
            if not res.ok:return res
            data=json.loads(report_path.read_text(encoding="utf-8"));report=self.parse_report(data)
            return ToolResult.success(self.name,"DRC clean" if report.drc_clean else f"DRC found {report.error_count} errors",artifacts={"drc_path":str(report_path),"report":report.model_dump(mode="json")},metrics={"drc_clean":report.drc_clean,"error_count":report.error_count,"warning_count":report.warning_count},raw=res.raw)
        except Exception as exc:return ToolResult.failure(self.name,str(exc))

    @classmethod
    def parse_report(cls,data:dict[str,Any])->DRCReport:
        rows=[]
        for key in ("violations","items","drc_violations"):
            if isinstance(data.get(key),list):rows.extend(data[key])
        if isinstance(data.get("sheets"),list):
            for sheet in data["sheets"]:
                if isinstance(sheet,dict):
                    for key in ("violations","items"):
                        if isinstance(sheet.get(key),list):rows.extend(sheet[key])
        violations=[]
        for row in rows:
            if not isinstance(row,dict):continue
            msg=str(row.get("description") or row.get("message") or row.get("type") or "")
            sev=str(row.get("severity") or "error").lower();cat=cls._category(msg+" "+str(row.get("type") or ""))
            refs=cls._collect(row,("ref","reference","footprint"));nets=cls._collect(row,("net","net_name"));loc=cls._location(row)
            violations.append(DRCViolation(category=cat,severity=sev,message=msg,refs=refs,nets=nets,location_mm=loc,raw=row))
        errors=sum(v.severity not in {"warning","info","excluded"} for v in violations);warnings=sum(v.severity=="warning" for v in violations)
        # Some KiCad versions expose aggregate counts separately.
        errors=max(errors,int(data.get("error_count",0) or 0));warnings=max(warnings,int(data.get("warning_count",0) or 0))
        return DRCReport(drc_clean=errors==0,error_count=errors,warning_count=warnings,violations=violations,raw=data)
    @staticmethod
    def _category(text:str)->str:
        t=text.lower()
        if "clearance" in t:return "clearance"
        if "unconnected" in t or "unrouted" in t:return "unconnected"
        if "courtyard" in t:return "courtyard"
        if "keepout" in t or "rule area" in t:return "keepout"
        if "hole" in t or "drill" in t:return "hole_mechanical"
        if "schematic" in t:return "schematic_parity"
        return "other"
    @staticmethod
    def _collect(row:dict[str,Any],keys:tuple[str,...])->list[str]:
        out=[]
        def walk(v:Any):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k in keys and x not in (None,""):out.append(str(x))
                    walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(row);return sorted(set(out))
    @staticmethod
    def _location(row:dict[str,Any])->tuple[float,float]|None:
        for key in ("position","location","pos"):
            v=row.get(key)
            if isinstance(v,dict) and "x" in v and "y" in v:return (float(v["x"]),float(v["y"]))
            if isinstance(v,(list,tuple)) and len(v)>=2:return (float(v[0]),float(v[1]))
        return None
