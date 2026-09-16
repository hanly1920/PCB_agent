from __future__ import annotations
import argparse,json,logging,sys
from pathlib import Path
from ..agent import AgentOrchestrator
from ..config import AgentConfig
from ..llm import DSLCompiler
from ..tools.drc_kicad import KicadDRCTool

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog="pcb-agent");sub=p.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run")
    run.add_argument("--bundle",help="Design bundle directory or manifest; kept for backward compatibility")
    run.add_argument("--input",help="Design bundle directory, bundle manifest, or .kicad_pcb board")
    run.add_argument("--board",help="Alias for --input when passing a .kicad_pcb board")
    run.add_argument("--out",required=True)
    run.add_argument("--config")
    run.add_argument("--text",default="")
    run.add_argument("--ckpt")
    run.add_argument("--device")
    run.add_argument("--mock",action="store_true",help="Use deterministic mock placement/router/DRC to verify the full workflow")
    run.add_argument("--skip-routing",action="store_true",help="Generate placement without running the router")
    run.add_argument("--skip-drc",action="store_true",help="Generate placement without running KiCad DRC")
    run.add_argument("--require-drc-clean",action="store_true",help="Return success only when DRC is clean")
    run.add_argument("--no-tool-fallback",action="store_true",help="Fail instead of falling back when router/DRC tools are unavailable")
    run.add_argument("--no-llm-tuning",action="store_true",help="Disable LLM layout critic and use rule repair fallback only")
    run.add_argument("--no-online-finetune",action="store_true",help="Disable online policy-gradient/self-imitation checkpoint updates")
    comp=sub.add_parser("compile-dsl");comp.add_argument("--text",required=True);comp.add_argument("--config")
    drc=sub.add_parser("drc");drc.add_argument("--board",required=True);drc.add_argument("--out",required=True);drc.add_argument("--config")
    val=sub.add_parser("validate-checkpoint");val.add_argument("--ckpt",required=True)
    args=p.parse_args(argv);logging.basicConfig(level=logging.INFO,format="%(levelname)s %(name)s: %(message)s")
    if args.command=="compile-dsl":
        cfg=AgentConfig.load(args.config);print(json.dumps(DSLCompiler(cfg.llm).compile(args.text).model_dump(mode="json"),ensure_ascii=False,indent=2));return 0
    if args.command=="validate-checkpoint":
        from ..config import PlacementConfig
        from ..tools import PlacementModelTool
        print(json.dumps(PlacementModelTool(PlacementConfig(checkpoint_path=Path(args.ckpt))).validate_checkpoint(),ensure_ascii=False,indent=2));return 0
    cfg=AgentConfig.load(args.config)
    if args.command=="drc":
        result=KicadDRCTool(cfg.kicad).run(args.board,args.out);print(json.dumps(result.model_dump(mode="json"),ensure_ascii=False,indent=2));return 0 if result.ok else 2
    if args.ckpt:cfg.placement.checkpoint_path=Path(args.ckpt)
    if args.device:cfg.placement.device=args.device
    if args.mock:cfg.placement.mock=True;cfg.router.mode="mock";cfg.kicad.mock_drc=True;cfg.llm.provider="mock"
    if args.skip_routing:cfg.workflow.run_routing=False
    if args.skip_drc:cfg.workflow.run_drc=False
    if args.require_drc_clean:cfg.workflow.require_drc_clean=True;cfg.workflow.accept_placement_without_drc=False
    if args.no_tool_fallback:cfg.workflow.auto_skip_unavailable_tools=False
    if args.no_llm_tuning:cfg.workflow.llm_tuning_enabled=False
    if args.no_online_finetune:cfg.online_finetune.enabled=False
    source=args.input or args.board or args.bundle
    if not source:
        p.error("run requires --input, --board, or --bundle")
    state=AgentOrchestrator(cfg).run(source,args.out,args.text);print(json.dumps({"status":state.status,"board_path":str(state.board_path) if state.board_path else None,"metrics":state.metrics},ensure_ascii=False,indent=2));return 0 if state.status=="EXPORT_RESULT" else 2

if __name__=="__main__":raise SystemExit(main())
