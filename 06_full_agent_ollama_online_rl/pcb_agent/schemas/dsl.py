from __future__ import annotations
from typing import Any, Literal
from pydantic import Field, field_validator
from .base import AgentModel

class Constraint(AgentModel):
    type: str
    refs: list[str] = Field(default_factory=list)
    target: str | None = None
    priority: Literal["hard", "soft"] | None = None
    weight: float | None = None
    side: str | None = None
    region: str | None = None
    group: str | None = None
    radius_mm: float | None = None
    value_mm: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target", mode="before")
    @classmethod
    def normalize_target(cls, value):
        if isinstance(value, list):
            return str(value[0]) if value else None
        return value

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("weight must be non-negative")
        return value

class IterationPolicy(AgentModel):
    max_placement_rounds: int = 3
    max_routing_rounds: int = 2
    candidate_count: int = 3
    repair_window_mm: float = 8.0
    stop_when_drc_clean: bool = True
    replay_temperature: float = 1.0
    rollout_budget: int = 16
    policy_updates: int = 0

class LayoutDSL(AgentModel):
    version: str = "0.1"
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_constraints: list[Constraint] = Field(default_factory=list)
    objective_weights: dict[str, float] = Field(default_factory=lambda: {
        "hpwl": 1.0,
        "weighted_hpwl": 0.2,
        "nslw": 0.1,
        "drc": 100.0,
        "unrouted": 50.0,
        "user_preference": 2.0,
    })
    locked_refs: list[str] = Field(default_factory=list)
    movable_refs: list[str] = Field(default_factory=list)
    region_constraints: list[Constraint] = Field(default_factory=list)
    edge_constraints: list[Constraint] = Field(default_factory=list)
    routing_constraints: dict[str, Any] = Field(default_factory=dict)
    iteration_policy: IterationPolicy = Field(default_factory=IterationPolicy)

    def objective_overrides(self) -> dict[str, float]:
        mapping = {
            "hpwl": "hpwl_weight",
            "weighted_hpwl": "w_hpwl_weight",
            "nslw": "nslw_weight",
            "alignment": "objective_align_weight",
            "density": "objective_density_weight",
            "spacing": "objective_soft_spacing_weight",
            "neatness": "objective_neatness_weight",
        }
        return {mapping[k]: float(v) for k, v in self.objective_weights.items() if k in mapping}
