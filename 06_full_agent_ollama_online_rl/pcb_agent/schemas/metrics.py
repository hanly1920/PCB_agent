from __future__ import annotations
from typing import Any
from pydantic import Field
from .base import AgentModel

class LayoutCandidate(AgentModel):
    candidate_id: str
    placements: dict[str, dict[str, Any]]
    objective: float = 1e12
    metrics: dict[str, Any] = Field(default_factory=dict)
    legal: bool = False
    complete: bool = False
    source: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict)

class DRCViolation(AgentModel):
    category: str = "other"
    severity: str = "error"
    message: str = ""
    refs: list[str] = Field(default_factory=list)
    nets: list[str] = Field(default_factory=list)
    location_mm: tuple[float, float] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

class DRCReport(AgentModel):
    drc_clean: bool = False
    error_count: int = 0
    warning_count: int = 0
    violations: list[DRCViolation] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
