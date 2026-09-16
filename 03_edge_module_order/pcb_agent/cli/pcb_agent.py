from __future__ import annotations
import argparse,json,logging,sys
from pathlib import Path
from ..agent import AgentOrchestrator
from ..config import AgentConfig
from ..llm import DSLCompiler
from ..tools.drc_kicad import KicadDRCTool

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog="pcb-agent");sub=p.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run");run.add_argument("--bundle",required=True);run.add_argument("--out",required=True);run.add_argument("--config");run.add_argument("--text",default="");run.add_argument("--ckpt");run.add_argument("--device");run.add_argument("--mock",action="store_true",help="Use deterministic mock placement/router/DRC to verify the full workflow")
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
    state=AgentOrchestrator(cfg).run(args.bundle,args.out,args.text);print(json.dumps({"status":state.status,"board_path":str(state.board_path) if state.board_path else None,"metrics":state.metrics},ensure_ascii=False,indent=2));return 0 if state.status=="EXPORT_RESULT" else 2

if __name__=="__main__":raise SystemExit(main())
