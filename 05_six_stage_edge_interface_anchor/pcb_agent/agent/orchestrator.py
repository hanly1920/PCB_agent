from __future__ import annotations
import json, logging, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ..config import AgentConfig
from ..llm import DSLCompiler
from ..schemas import LayoutDSL, ToolResult
from ..tools import DesignIngestTool, PlacementModelTool, LayoutAnalyzer, KicadBridge, FreeroutingRouter, KicadDRCTool, RepairPlanner
from .state import AgentState

log=logging.getLogger(__name__)
class AgentOrchestrator:
    def __init__(self,config:AgentConfig):
        self.config=config;self.ingest=DesignIngestTool();self.compiler=DSLCompiler(config.llm);self.placement=PlacementModelTool(config.placement);self.analyzer=LayoutAnalyzer();self.bridge=KicadBridge();self.router=FreeroutingRouter(config.router);self.drc=KicadDRCTool(config.kicad);self.repair=RepairPlanner()

    def run(self,bundle_dir:str|Path,output_dir:str|Path,user_text:str="") -> AgentState:
        out=Path(output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True);state=AgentState(run_id=out.name,checkpoint_path=self.config.placement.checkpoint_path)
        try:
            ing=self._call(state,"INGEST",self.ingest.run(bundle_dir,out));self._require(ing);state.board_path=Path(ing.artifacts["board_path"]);state.task_json_path=Path(ing.artifacts["task_json_path"]);bundle_data=ing.artifacts.get("bundle") or {};request=user_text or bundle_data.get("user_text") or ""
            state.status="COMPILE_DSL";state.dsl=self.compiler.compile(request,{"board_path":str(state.board_path),"round":0});self._event(state,"COMPILE_DSL",True,"DSL compiled",{"dsl":state.dsl.model_dump(mode="json")});self._save(state,out)
            max_rounds=min(max(1,self.config.max_rounds),max(1,state.dsl.iteration_policy.max_placement_rounds))
            for round_index in range(max_rounds):
                state.round_index=round_index
                place=self._call(state,"GENERATE_PLACEMENT_CANDIDATES",self.placement.run(state.task_json_path,state.dsl));self._require(place);state.candidates=place.artifacts["candidates"]
                analysis=self._call(state,"ANALYZE_PLACEMENTS",self.analyzer.run(state.candidates,state.task_json_path,state.dsl));self._require(analysis);state.selected_candidate=analysis.artifacts["best_candidate"];state.metrics.update(analysis.metrics)
                placed_board=out/f"round_{round_index:02d}"/"placed.kicad_pcb";apply=self._call(state,"APPLY_BEST_TO_KICAD",self.bridge.apply_placement(state.board_path,state.selected_candidate["placements"],placed_board));self._require(apply)
                route=self._call(state,"AUTOROUTE",self.router.run(placed_board,placed_board.parent));self._require(route);routed=Path(route.artifacts["routed_board_path"])
                drc=self._call(state,"RUN_DRC",self.drc.run(routed,placed_board.parent));self._require(drc);state.drc_report=drc.artifacts["report"];state.metrics.update(drc.metrics);state.board_path=routed
                if bool(drc.metrics.get("drc_clean")):
                    state.status="EXPORT_RESULT";final=out/"final.kicad_pcb";shutil.copy2(routed,final);state.board_path=final;self._event(state,"EXPORT_RESULT",True,"Final board exported",{"board_path":str(final)});break
                if round_index+1>=max_rounds:
                    state.status="REPORT_FAILURE";self._event(state,"REPORT_FAILURE",False,"Maximum rounds reached",{});break
                repair=self._call(state,"DIAGNOSE_AND_REPAIR",self.repair.run(state.drc_report,state.metrics,state.dsl));self._require(repair);self._apply_patch(state.dsl,repair.artifacts["plan"].get("dsl_patch") or {})
            self._write_report(state,out);self._save(state,out);return state
        except Exception as exc:
            state.status="FAILED";self._event(state,"FAILED",False,str(exc),{});self._write_report(state,out);self._save(state,out);return state

    def _call(self,state:AgentState,status:str,result:ToolResult)->ToolResult:
        state.status=status;self._event(state,status,result.ok,result.summary,result.model_dump(mode="json"));return result
    @staticmethod
    def _require(result:ToolResult)->None:
        if not result.ok:raise RuntimeError("; ".join(result.errors) or result.summary)
    @staticmethod
    def _apply_patch(dsl:LayoutDSL,patch:dict[str,Any])->None:
        for k,v in (patch.get("objective_weights") or {}).items():dsl.objective_weights[k]=float(v)
    def _event(self,state:AgentState,status:str,ok:bool,summary:str,payload:dict[str,Any])->None:
        event={"time":datetime.now(timezone.utc).isoformat(),"status":status,"ok":ok,"summary":summary,"round":state.round_index,"payload":payload};state.history.append(event);state.updated_at=event["time"]
    @staticmethod
    def _save(state: AgentState, out: Path) -> None:
        (out / "state.json").write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (out / "run_history.jsonl").open("w", encoding="utf-8") as handle:
            for event in state.history:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        (out / "metrics.json").write_text(
            json.dumps(state.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_report(state: AgentState, out: Path) -> None:
        clean = bool((state.drc_report or {}).get("drc_clean"))
        lines = [
            "# PCB Agent Run Report",
            "",
            f"- Run ID: `{state.run_id}`",
            f"- Status: `{state.status}`",
            f"- Rounds: `{state.round_index + 1}`",
            f"- DRC clean: `{clean}`",
            f"- Final board: `{state.board_path}`",
            "",
            "## Metrics",
            "```json",
            json.dumps(state.metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Notes",
            "- Real placement requires CUDA because the bundled `pcbplace.infer_layout` enforces CUDA.",
            "- Real Freerouting mode requires configured DSN export and SES import adapter commands.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
