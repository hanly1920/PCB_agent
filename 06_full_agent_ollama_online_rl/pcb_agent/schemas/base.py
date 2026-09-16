from __future__ import annotations
from pydantic import BaseModel, ConfigDict

class AgentModel(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
