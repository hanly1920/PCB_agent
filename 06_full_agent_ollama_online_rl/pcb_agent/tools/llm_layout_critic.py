from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import LLMConfig
from ..llm.qwen_client import QwenClient
from ..schemas import LayoutDSL, ToolResult, TuningPlan
from .repair import RepairPlanner

_JSON_RE = re.compile(r"\{.*\}", re.S)

CRITIC_SYSTEM_PROMPT = """You are a PCB layout critic and replay tuning planner.
You must analyze deterministic placement metrics, candidate rankings, routing/DRC reports, and user constraints.
Return exactly one JSON object that validates against the supplied TuningPlan schema.
Never output component coordinates, absolute placement positions, KiCad text edits, or prose outside JSON.
You may only tune DSL constraints, objective weights, replay/rollout parameters, and freeze/release reference sets.
Prefer local_replay when violations identify component references; prefer global_replay when the failure is diffuse congestion or no references are available.
Use accept only when the layout is complete, legal, satisfies hard constraints, and DRC is clean or unavailable by configuration.
"""


class LLMLayoutCritic:
    name = "llm_layout_critic"

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = QwenClient(config)
        self.fallback = RepairPlanner()

    def run(
        self,
        *,
        task_json_path: str | Path,
        dsl: LayoutDSL,
        candidates: list[dict[str, Any]],
        ranking: list[dict[str, Any]],
        selected_candidate: dict[str, Any] | None,
        metrics: dict[str, Any],
        drc_report: dict[str, Any] | None,
        round_index: int,
        user_text: str = "",
    ) -> ToolResult:
        try:
            summary = self._build_summary(
                task_json_path=Path(task_json_path),
                dsl=dsl,
                candidates=candidates,
                ranking=ranking,
                selected_candidate=selected_candidate or {},
                metrics=metrics,
                drc_report=drc_report or {},
                round_index=round_index,
                user_text=user_text,
            )
            if self.config.provider in {"mock", "disabled"}:
                return self._fallback(summary, dsl, drc_report or {}, metrics, reason=f"provider={self.config.provider}")
            raw = self.client.chat(
                [
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"summary": summary, "schema": TuningPlan.model_json_schema()}, ensure_ascii=False)},
                ],
                json_schema=TuningPlan.model_json_schema(),
            )
            plan = TuningPlan.model_validate(self._extract_json(raw))
            plan = self._sanitize_plan(plan, summary)
            return ToolResult.success(
                self.name,
                "LLM tuning plan generated",
                artifacts={"plan": plan.model_dump(mode="json"), "critic_summary": summary},
                raw={"llm_raw": raw},
            )
        except Exception as exc:
            return self._fallback({}, dsl, drc_report or {}, metrics, reason=str(exc))

    def _fallback(self, summary: dict[str, Any], dsl: LayoutDSL, drc_report: dict[str, Any], metrics: dict[str, Any], *, reason: str) -> ToolResult:
        repair = self.fallback.run(drc_report, metrics, dsl)
        if repair.ok:
            legacy = repair.artifacts.get("plan") or {}
            plan = TuningPlan(
                action="local_replay" if legacy.get("target_refs") else "global_replay",
                diagnosis=f"Fallback rule planner used because LLM critic was unavailable: {reason}",
                target_refs=list(legacy.get("target_refs") or []),
                freeze_other_refs=bool(legacy.get("freeze_other_refs", True)),
                dsl_patch={"objective_weights": (legacy.get("dsl_patch") or {}).get("objective_weights") or {}, "movable_refs": (legacy.get("dsl_patch") or {}).get("movable_refs") or []},
                replay_patch={"candidate_count": max(3, int(getattr(dsl.iteration_policy, "candidate_count", 3))), "freeze_unrelated_refs": bool(legacy.get("freeze_other_refs", True))},
                expected_effect="Rule fallback increases objective pressure around reported violations.",
                confidence=0.35,
            )
        else:
            plan = TuningPlan(action="global_replay", diagnosis=f"No LLM or rule repair available: {reason}", freeze_other_refs=False, confidence=0.1)
        return ToolResult.success(
            self.name,
            "Fallback tuning plan generated",
            artifacts={"plan": plan.model_dump(mode="json"), "critic_summary": summary, "fallback_reason": reason},
            observations=["LLM critic fallback was used"],
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        raw = str(raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = _JSON_RE.search(raw)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _sanitize_plan(plan: TuningPlan, summary: dict[str, Any]) -> TuningPlan:
        # Defense-in-depth: forbid coordinate-like payloads in LLM params.
        dumped = plan.model_dump(mode="json")
        bad_keys = {"x", "y", "xy", "xy_mm", "position", "position_mm", "at", "at_mm", "rotation", "rot"}
        def walk(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    if str(k).lower() in bad_keys:
                        raise ValueError(f"LLM tuning plan attempted to emit coordinate-like key: {path}.{k}")
                    walk(v, f"{path}.{k}")
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    walk(v, f"{path}[{i}]")
        walk(dumped)
        known_refs = set(summary.get("component_refs") or [])
        if known_refs:
            plan.target_refs = sorted({r for r in plan.target_refs if r in known_refs})
            plan.dsl_patch.locked_refs = sorted({r for r in plan.dsl_patch.locked_refs if r in known_refs})
            plan.dsl_patch.movable_refs = sorted({r for r in plan.dsl_patch.movable_refs if r in known_refs})
        return plan

    @staticmethod
    def _build_summary(
        *,
        task_json_path: Path,
        dsl: LayoutDSL,
        candidates: list[dict[str, Any]],
        ranking: list[dict[str, Any]],
        selected_candidate: dict[str, Any],
        metrics: dict[str, Any],
        drc_report: dict[str, Any],
        round_index: int,
        user_text: str,
    ) -> dict[str, Any]:
        task: dict[str, Any] = {}
        try:
            task = json.loads(task_json_path.read_text(encoding="utf-8"))
        except Exception:
            task = {}
        comps = task.get("components") or []
        refs = [str(c.get("ref")) for c in comps if c.get("ref")]
        fixed_refs = [str(c.get("ref")) for c in comps if c.get("ref") and (c.get("fixed") or c.get("locked"))]
        board = task.get("board") or {}
        top_ranking = []
        for row in (ranking or [])[:8]:
            top_ranking.append({
                "candidate_id": row.get("candidate_id"),
                "score": row.get("score"),
                "soft_score": row.get("soft_score"),
                "hard_violation_count": row.get("hard_violation_count"),
                "hard_violations": row.get("hard_violations"),
                "legal": row.get("legal"),
                "complete": row.get("complete"),
                "objective": row.get("objective"),
            })
        violations = []
        for v in (drc_report.get("violations") or [])[:30]:
            if isinstance(v, dict):
                violations.append({
                    "category": v.get("category"),
                    "severity": v.get("severity"),
                    "message": v.get("message"),
                    "refs": v.get("refs"),
                    "nets": v.get("nets"),
                    "location_mm": v.get("location_mm"),
                })
        selected_metrics = dict(selected_candidate.get("metrics") or {})
        return {
            "round_index": int(round_index),
            "user_text": user_text,
            "board": board,
            "component_count": len(refs),
            "component_refs": refs[:500],
            "fixed_refs": fixed_refs[:200],
            "dsl": dsl.model_dump(mode="json"),
            "candidate_count": len(candidates or []),
            "ranking_top": top_ranking,
            "selected_candidate_id": selected_candidate.get("candidate_id"),
            "selected_complete": selected_candidate.get("complete"),
            "selected_legal": selected_candidate.get("legal"),
            "selected_objective": selected_candidate.get("objective"),
            "selected_metrics": selected_metrics,
            "agent_metrics": metrics,
            "drc": {
                "available": metrics.get("drc_available"),
                "clean": drc_report.get("drc_clean"),
                "error_count": drc_report.get("error_count"),
                "warning_count": drc_report.get("warning_count"),
                "violations": violations,
            },
            "routing": {
                "available": metrics.get("routing_available"),
                "error": metrics.get("routing_error"),
            },
            "instruction": "Return tuning only: no coordinates. Tune objective_weights, constraints, candidate_count, beam_width, beam_topk, replay_temperature, rollout_budget, policy_updates, and freeze/release refs.",
        }
