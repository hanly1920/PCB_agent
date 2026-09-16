from __future__ import annotations
from pathlib import Path
from ..schemas import ToolResult
class PreviewTool:
    name="preview"
    def run(self,board_path:str|Path,output_dir:str|Path)->ToolResult:
        return ToolResult.success(self.name,"Preview adapter reserved",artifacts={"board_path":str(board_path)},observations=["Use KiCad GUI or configure a plotting adapter for rendered previews"])
