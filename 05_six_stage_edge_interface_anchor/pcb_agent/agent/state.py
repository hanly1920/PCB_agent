from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import Field
from ..schemas.base import AgentModel
from ..schemas.dsl import LayoutDSL

class AgentState(AgentModel):
    run_id:str
    status:str="INITIALIZED"
    round_index:int=0
    board_path:Path|None=None
    task_json_path:Path|None=None
    dsl:LayoutDSL|None=None
    checkpoint_path:Path|None=None
    candidates:list[dict[str,Any]]=Field(default_factory=list)
    selected_candidate:dict[str,Any]|None=None
    drc_report:dict[str,Any]|None=None
    metrics:dict[str,Any]=Field(default_factory=dict)
    history:list[dict[str,Any]]=Field(default_factory=list)
    created_at:str=Field(default_factory=lambda:datetime.now(timezone.utc).isoformat())
    updated_at:str=Field(default_factory=lambda:datetime.now(timezone.utc).isoformat())
