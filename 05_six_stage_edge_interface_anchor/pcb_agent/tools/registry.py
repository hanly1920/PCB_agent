from __future__ import annotations
from typing import Any
class ToolRegistry:
    def __init__(self):self._tools:dict[str,Any]={}
    def register(self,name:str,tool:Any)->None:self._tools[name]=tool
    def get(self,name:str)->Any:return self._tools[name]
    def names(self)->list[str]:return sorted(self._tools)
