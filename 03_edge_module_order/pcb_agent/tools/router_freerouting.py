from __future__ import annotations
import shutil
from pathlib import Path
from ..config import RouterConfig
from ..schemas import ToolResult
from .command import CommandRunner

class FreeroutingRouter:
    name="freerouting_router"
    def __init__(self,config:RouterConfig):self.config=config;self.runner=CommandRunner()
    @staticmethod
    def _render(parts:list[str],mapping:dict[str,str])->list[str]:return [p.format(**mapping) for p in parts]
    def run(self,placed_board:str|Path,output_dir:str|Path)->ToolResult:
        try:
            board=Path(placed_board);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);routed=out/"routed.kicad_pcb"
            if self.config.mode in {"mock","none"}:
                shutil.copy2(board,routed)
                return ToolResult.success(self.name,"Routing bypassed" if self.config.mode=="none" else "Mock routing completed",artifacts={"routed_board_path":str(routed)},metrics={"mock":self.config.mode=="mock","unrouted_nets":0 if self.config.mode=="mock" else None})
            if not self.config.freerouting_jar or not Path(self.config.freerouting_jar).exists():raise FileNotFoundError("freerouting_jar is missing")
            if not self.config.export_dsn_cmd or not self.config.import_ses_cmd:raise RuntimeError("Real Freerouting mode requires export_dsn_cmd and import_ses_cmd adapters")
            dsn=out/"board.dsn";ses=out/"board.ses";mapping={"board":str(board),"dsn":str(dsn),"ses":str(ses),"output_board":str(routed)}
            exp=self.runner.run("export_dsn",self._render(self.config.export_dsn_cmd,mapping),timeout_sec=self.config.timeout_sec)
            if not exp.ok:return exp
            route=self.runner.run(self.name,[self.config.java_bin,"-jar",str(self.config.freerouting_jar),"-de",str(dsn),"-do",str(ses),"-mp",str(self.config.passes),"-mt",str(self.config.threads)],timeout_sec=self.config.timeout_sec)
            if not route.ok:return route
            imp=self.runner.run("import_ses",self._render(self.config.import_ses_cmd,mapping),timeout_sec=self.config.timeout_sec)
            if not imp.ok:return imp
            return ToolResult.success(self.name,"Freerouting completed",artifacts={"dsn_path":str(dsn),"ses_path":str(ses),"routed_board_path":str(routed)},raw={"export":exp.raw,"route":route.raw,"import":imp.raw})
        except Exception as exc:return ToolResult.failure(self.name,str(exc))
