from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, field_validator
from .base import AgentModel
from .dsl import Constraint


class DSLPatch(AgentModel):
    objective_weights: dict[str, float] = Field(default_factory=dict)
    locked_refs: list[str] = Field(default_factory=list)
    movable_refs: list[str] = Field(default_factory=list)
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_constraints: list[Constraint] = Field(default_factory=list)
    region_constraints: list[Constraint] = Field(default_factory=list)
    edge_constraints: list[Constraint] = Field(default_factory=list)


class ReplayPatch(AgentModel):
    candidate_count: int | None = None
    beam_width: int | None = None
    beam_topk: int | None = None
    layout_preset: str | None = None
    replay_temperature: float | None = None
    rollout_budget: int | None = None
    policy_updates: int | None = None
    freeze_unrelated_refs: bool | None = None

    @field_validator("candidate_count", "beam_width", "beam_topk", "rollout_budget")
    @classmethod
    def positive_int(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("replay integer fields must be positive")
        return value

    @field_validator("policy_updates")
    @classmethod
    def nonnegative_updates(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("policy_updates must be non-negative")
        return value

    @field_validator("replay_temperature")
    @classmethod
    def positive_temp(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("replay_temperature must be positive")
        return value


class TuningPlan(AgentModel):
    version: str = "0.1"
    action: Literal["accept", "local_replay", "global_replay", "request_more_info"] = "local_replay"
    diagnosis: str = ""
    target_refs: list[str] = Field(default_factory=list)
    freeze_other_refs: bool = True
    dsl_patch: DSLPatch = Field(default_factory=DSLPatch)
    replay_patch: ReplayPatch = Field(default_factory=ReplayPatch)
    expected_effect: str = ""
    stop_reason: str | None = None
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
