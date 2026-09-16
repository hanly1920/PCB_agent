from __future__ import annotations
from typing import Any
from pydantic import Field
from .base import AgentModel

class ToolResult(AgentModel):
    ok: bool
    name: str
    summary: str = ""
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, name: str, summary: str = "", **kwargs: Any) -> "ToolResult":
        return cls(ok=True, name=name, summary=summary, **kwargs)

    @classmethod
    def failure(cls, name: str, error: str, summary: str = "", **kwargs: Any) -> "ToolResult":
        errors = list(kwargs.pop("errors", []))
        errors.append(str(error))
        return cls(ok=False, name=name, summary=summary or str(error), errors=errors, **kwargs)
