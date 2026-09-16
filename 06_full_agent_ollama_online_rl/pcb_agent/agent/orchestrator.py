from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..llm import DSLCompiler
from ..schemas import LayoutDSL, ToolResult, TuningPlan
from ..tools import (
    DesignIngestTool,
    FreeroutingRouter,
    KicadBridge,
    KicadDRCTool,
    LayoutAnalyzer,
    LLMLayoutCritic,
    OnlinePolicyFinetuner,
    PlacementModelTool,
    RepairPlanner,
)
from .state import AgentState

log = logging.getLogger(__name__)


class AgentOrchestrator:
    """End-to-end PCB placement agent.

    The orchestrator is intentionally deterministic and replayable: every tool
    result is recorded, every selected candidate is written to disk, and missing
    installation-specific tools such as Freerouting adapters or KiCad DRC can be
    skipped only when the workflow config explicitly allows automatic fallback.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.ingest = DesignIngestTool()
        self.compiler = DSLCompiler(config.llm)
        self.placement = PlacementModelTool(config.placement)
        self.analyzer = LayoutAnalyzer()
        self.bridge = KicadBridge()
        self.router = FreeroutingRouter(config.router)
        self.drc = KicadDRCTool(config.kicad)
        self.repair = RepairPlanner()
        self.critic = LLMLayoutCritic(config.llm)
        self.online_finetuner = OnlinePolicyFinetuner(config.online_finetune, config.placement)

    def run(self, bundle_dir: str | Path, output_dir: str | Path, user_text: str = "") -> AgentState:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        state = AgentState(run_id=out.name, checkpoint_path=self.config.placement.checkpoint_path)
        original_board: Path | None = None
        try:
            ing = self._call(state, "INGEST", self.ingest.run(bundle_dir, out))
            self._require(ing)
            state.board_path = Path(ing.artifacts["board_path"])
            original_board = state.board_path
            state.task_json_path = Path(ing.artifacts["task_json_path"])
            bundle_data = ing.artifacts.get("bundle") or {}
            request = user_text or bundle_data.get("user_text") or ""

            state.status = "COMPILE_DSL"
            state.dsl = self.compiler.compile(request, {"board_path": str(state.board_path), "round": 0})
            self._event(state, "COMPILE_DSL", True, "DSL compiled", {"dsl": state.dsl.model_dump(mode="json")})
            self._save(state, out)

            max_rounds = min(max(1, self.config.max_rounds), max(1, state.dsl.iteration_policy.max_placement_rounds))
            for round_index in range(max_rounds):
                state.round_index = round_index
                round_dir = out / f"round_{round_index:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)

                place = self._call(
                    state,
                    "GENERATE_PLACEMENT_CANDIDATES",
                    self.placement.run(state.task_json_path, state.dsl),
                )
                self._require(place)
                state.candidates = place.artifacts["candidates"]
                self._write_json(round_dir / "candidates.json", state.candidates)

                analysis_task_path = Path(place.artifacts.get("task_json_path") or state.task_json_path)
                analysis = self._call(
                    state,
                    "ANALYZE_PLACEMENTS",
                    self.analyzer.run(state.candidates, analysis_task_path, state.dsl),
                )
                self._require(analysis)
                state.selected_candidate = analysis.artifacts["best_candidate"]
                state.metrics.update(analysis.metrics)
                ranking = analysis.artifacts.get("ranking") or []
                self._write_json(round_dir / "candidate_ranking.json", ranking)
                self._write_json(round_dir / "selected_candidate.json", state.selected_candidate)
                if ranking:
                    state.metrics["best_hard_violations"] = ranking[0].get("hard_violations") or []

                placed_board = round_dir / "placed.kicad_pcb"
                apply = self._call(
                    state,
                    "APPLY_BEST_TO_KICAD",
                    self.bridge.apply_placement(original_board or state.board_path, state.selected_candidate["placements"], placed_board),
                )
                self._require(apply)
                current_board = placed_board

                routed_board = self._route_if_available(state, current_board, round_dir)
                if routed_board is not None:
                    current_board = routed_board

                drc_available = False
                if self.config.workflow.run_drc:
                    drc_result = self.drc.run(current_board, round_dir)
                    if drc_result.ok:
                        drc = self._call(state, "RUN_DRC", drc_result)
                        state.drc_report = drc.artifacts["report"]
                        state.metrics.update(drc.metrics)
                        state.metrics["drc_available"] = True
                        drc_available = True
                    elif self.config.workflow.auto_skip_unavailable_tools:
                        self._event(
                            state,
                            "RUN_DRC_SKIPPED",
                            True,
                            "DRC unavailable; continuing with placement-level acceptance",
                            drc_result.model_dump(mode="json"),
                        )
                        state.drc_report = {"drc_clean": False, "unavailable": True, "errors": drc_result.errors}
                        state.metrics.update({"drc_available": False, "drc_clean": False})
                    else:
                        self._require(drc_result)
                else:
                    self._event(state, "RUN_DRC_SKIPPED", True, "DRC disabled by workflow config", {})
                    state.drc_report = {"drc_clean": False, "disabled": True}
                    state.metrics.update({"drc_available": False, "drc_clean": False})

                accepted, reason = self._acceptance_decision(state, drc_available)
                min_rounds = max(1, int(self.config.workflow.min_optimization_rounds))
                if accepted and (round_index + 1) >= min_rounds:
                    self._export_result(state, current_board, out, reason)
                    break

                if round_index + 1 >= max_rounds:
                    if self.config.workflow.export_best_effort_board and self._candidate_is_usable(state):
                        self._export_result(state, current_board, out, "best_effort_after_max_rounds")
                    else:
                        state.status = "REPORT_FAILURE"
                        self._event(state, "REPORT_FAILURE", False, "Maximum rounds reached without an acceptable layout", {})
                    break

                if self.config.workflow.llm_tuning_enabled:
                    critic = self._call(
                        state,
                        "LLM_LAYOUT_CRITIC_TUNE",
                        self.critic.run(
                            task_json_path=analysis_task_path,
                            dsl=state.dsl,
                            candidates=state.candidates,
                            ranking=ranking,
                            selected_candidate=state.selected_candidate,
                            metrics=state.metrics,
                            drc_report=state.drc_report,
                            round_index=round_index,
                            user_text=request,
                        ),
                    )
                    self._require(critic)
                    plan = critic.artifacts["plan"]
                else:
                    repair = self._call(
                        state,
                        "DIAGNOSE_AND_REPAIR",
                        self.repair.run(state.drc_report or {}, state.metrics, state.dsl),
                    )
                    self._require(repair)
                    legacy = repair.artifacts["plan"]
                    plan = {
                        "action": legacy.get("action", "local_replay"),
                        "diagnosis": legacy.get("reason", "rule repair planner"),
                        "target_refs": legacy.get("target_refs") or [],
                        "freeze_other_refs": bool(legacy.get("freeze_other_refs", True)),
                        "dsl_patch": legacy.get("dsl_patch") or {},
                        "replay_patch": {"freeze_unrelated_refs": bool(legacy.get("freeze_other_refs", True))},
                    }

                tuning_plan = TuningPlan.model_validate(plan)
                self._write_json(round_dir / "tuning_plan.json", tuning_plan.model_dump(mode="json"))
                if tuning_plan.action == "accept" and self.config.workflow.allow_llm_accept and self._candidate_is_usable(state):
                    self._export_result(state, current_board, out, "llm_accept")
                    break
                self._apply_tuning_plan(state.dsl, tuning_plan)
                replay_task_plan = self._replay_task_plan(tuning_plan)
                state.task_json_path = self._write_replay_task(
                    analysis_task_path,
                    state.selected_candidate,
                    replay_task_plan,
                    round_dir / "replay_task_next.json",
                )
                policy_updates = int(
                    tuning_plan.replay_patch.policy_updates
                    if tuning_plan.replay_patch.policy_updates is not None
                    else (state.dsl.iteration_policy.policy_updates or self.config.online_finetune.default_policy_updates)
                )
                finetune = self._call(
                    state,
                    "ONLINE_POLICY_FINETUNE",
                    self.online_finetuner.run(
                        task_json_path=state.task_json_path,
                        selected_candidate=state.selected_candidate,
                        dsl=state.dsl,
                        output_dir=out,
                        round_index=round_index,
                        policy_updates=policy_updates,
                    ),
                )
                self._require(finetune)
                new_checkpoint = finetune.artifacts.get("checkpoint_path")
                if new_checkpoint:
                    self.config.placement.checkpoint_path = Path(new_checkpoint)
                    state.checkpoint_path = Path(new_checkpoint)
                    state.metrics["active_checkpoint_path"] = str(new_checkpoint)
                self._save(state, out)

            self._write_report(state, out)
            self._save(state, out)
            return state
        except Exception as exc:
            state.status = "FAILED"
            self._event(state, "FAILED", False, str(exc), {})
            self._write_report(state, out)
            self._save(state, out)
            return state

    def _route_if_available(self, state: AgentState, placed_board: Path, round_dir: Path) -> Path | None:
        if not self.config.workflow.run_routing or self.config.router.mode == "none":
            self._event(state, "AUTOROUTE_SKIPPED", True, "Routing disabled by workflow config", {})
            state.metrics["routing_available"] = False
            return None
        route_result = self.router.run(placed_board, round_dir)
        if route_result.ok:
            route = self._call(state, "AUTOROUTE", route_result)
            state.metrics["routing_available"] = True
            return Path(route.artifacts["routed_board_path"])
        if self.config.workflow.auto_skip_unavailable_tools:
            self._event(
                state,
                "AUTOROUTE_SKIPPED",
                True,
                "Routing unavailable; continuing with unrouted placed board",
                route_result.model_dump(mode="json"),
            )
            state.metrics["routing_available"] = False
            state.metrics["routing_error"] = "; ".join(route_result.errors) or route_result.summary
            return None
        self._require(route_result)
        return None

    def _acceptance_decision(self, state: AgentState, drc_available: bool) -> tuple[bool, str]:
        if bool((state.drc_report or {}).get("drc_clean")):
            return True, "drc_clean"
        if self.config.workflow.require_drc_clean:
            return False, "drc_required"
        if not drc_available and self.config.workflow.accept_placement_without_drc and self._candidate_is_usable(state):
            return True, "placement_accepted_without_drc"
        return False, "needs_repair"

    @staticmethod
    def _candidate_is_usable(state: AgentState) -> bool:
        cand = state.selected_candidate or {}
        if not (bool(cand.get("complete")) and bool(cand.get("legal"))):
            return False
        hard = state.metrics.get("best_hard_violations") or []
        return len(hard) == 0

    def _export_result(self, state: AgentState, board: Path, out: Path, reason: str) -> None:
        state.status = "EXPORT_RESULT"
        final = out / "final.kicad_pcb"
        shutil.copy2(board, final)
        state.board_path = final
        state.metrics["acceptance_reason"] = reason
        state.metrics["final_board_path"] = str(final)
        self._event(state, "EXPORT_RESULT", True, f"Final board exported ({reason})", {"board_path": str(final)})

    def _call(self, state: AgentState, status: str, result: ToolResult) -> ToolResult:
        state.status = status
        self._event(state, status, result.ok, result.summary, result.model_dump(mode="json"))
        return result

    @staticmethod
    def _require(result: ToolResult) -> None:
        if not result.ok:
            raise RuntimeError("; ".join(result.errors) or result.summary)

    @staticmethod
    def _apply_patch(dsl: LayoutDSL, patch: dict[str, Any]) -> None:
        for k, v in (patch.get("objective_weights") or {}).items():
            dsl.objective_weights[k] = float(v)
        if "movable_refs" in patch:
            dsl.movable_refs = sorted({str(r) for r in (patch.get("movable_refs") or []) if str(r).strip()})
        if "locked_refs" in patch:
            dsl.locked_refs = sorted({str(r) for r in (patch.get("locked_refs") or []) if str(r).strip()})

    def _apply_tuning_plan(self, dsl: LayoutDSL, plan: TuningPlan) -> None:
        patch = plan.dsl_patch
        for k, v in (patch.objective_weights or {}).items():
            dsl.objective_weights[str(k)] = float(v)
        if patch.locked_refs:
            dsl.locked_refs = sorted({*map(str, dsl.locked_refs), *map(str, patch.locked_refs)})
        if patch.movable_refs:
            dsl.movable_refs = sorted({str(r) for r in patch.movable_refs if str(r).strip()})
        if patch.hard_constraints:
            dsl.hard_constraints.extend(patch.hard_constraints)
        if patch.soft_constraints:
            dsl.soft_constraints.extend(patch.soft_constraints)
        if patch.region_constraints:
            dsl.region_constraints.extend(patch.region_constraints)
            dsl.soft_constraints.extend(patch.region_constraints)
        if patch.edge_constraints:
            dsl.edge_constraints.extend(patch.edge_constraints)
            dsl.hard_constraints.extend(patch.edge_constraints)

        replay = plan.replay_patch
        if replay.candidate_count is not None:
            dsl.iteration_policy.candidate_count = int(replay.candidate_count)
        if replay.replay_temperature is not None:
            dsl.iteration_policy.replay_temperature = float(replay.replay_temperature)
        if replay.rollout_budget is not None:
            dsl.iteration_policy.rollout_budget = int(replay.rollout_budget)
        if replay.policy_updates is not None:
            dsl.iteration_policy.policy_updates = int(replay.policy_updates)
        if replay.beam_width is not None:
            self.config.placement.beam_width = int(replay.beam_width)
        if replay.beam_topk is not None:
            self.config.placement.beam_topk = int(replay.beam_topk)
        if replay.layout_preset:
            self.config.placement.layout_preset = str(replay.layout_preset)

    @staticmethod
    def _replay_task_plan(plan: TuningPlan) -> dict[str, Any]:
        targets = sorted({str(r) for r in (plan.target_refs or plan.dsl_patch.movable_refs or []) if str(r).strip()})
        freeze = bool(plan.freeze_other_refs)
        if plan.replay_patch.freeze_unrelated_refs is not None:
            freeze = bool(plan.replay_patch.freeze_unrelated_refs)
        if plan.action == "global_replay":
            freeze = False
        return {
            "action": plan.action,
            "target_refs": targets,
            "freeze_other_refs": freeze,
            "reason": plan.diagnosis,
            "dsl_patch": plan.dsl_patch.model_dump(mode="json"),
            "replay_patch": plan.replay_patch.model_dump(mode="json"),
        }

    @staticmethod
    def _write_replay_task(task_json_path: Path, selected_candidate: dict[str, Any], plan: dict[str, Any], output_path: Path) -> Path:
        if not plan.get("freeze_other_refs"):
            return task_json_path
        placements = selected_candidate.get("placements") or {}
        targets = {str(r) for r in (plan.get("target_refs") or [])}
        if not targets or not placements:
            return task_json_path
        data = json.loads(Path(task_json_path).read_text(encoding="utf-8"))
        frozen = 0
        released = 0
        for comp in data.get("components", []):
            ref = str(comp.get("ref") or "")
            if not ref:
                continue
            if ref in targets:
                if comp.get("fixed_source") == "agent_replay_freeze":
                    for key in ("fixed", "locked", "fixed_xy_mm", "fixed_rot", "fixed_source"):
                        comp.pop(key, None)
                    released += 1
                continue
            p = placements.get(ref)
            if not p:
                continue
            comp["fixed"] = True
            comp["locked"] = True
            comp["fixed_xy_mm"] = [float(p["x"]), float(p["y"])]
            comp["fixed_rot"] = float(p.get("rotation", 0.0))
            comp["fixed_source"] = "agent_replay_freeze"
            frozen += 1
        data.setdefault("meta", {})["agent_replay_plan"] = {**plan, "frozen_refs": frozen, "released_refs": released}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _event(self, state: AgentState, status: str, ok: bool, summary: str, payload: dict[str, Any]) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "ok": ok,
            "summary": summary,
            "round": state.round_index,
            "payload": payload,
        }
        state.history.append(event)
        state.updated_at = event["time"]

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _save(state: AgentState, out: Path) -> None:
        (out / "state.json").write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (out / "run_history.jsonl").open("w", encoding="utf-8") as handle:
            for event in state.history:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        (out / "metrics.json").write_text(
            json.dumps(state.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_report(state: AgentState, out: Path) -> None:
        clean = bool((state.drc_report or {}).get("drc_clean"))
        acceptance = state.metrics.get("acceptance_reason", "not_accepted")
        lines = [
            "# PCB Agent Run Report",
            "",
            f"- Run ID: `{state.run_id}`",
            f"- Status: `{state.status}`",
            f"- Rounds: `{state.round_index + 1}`",
            f"- Acceptance: `{acceptance}`",
            f"- DRC clean: `{clean}`",
            f"- Routing available: `{state.metrics.get('routing_available')}`",
            f"- DRC available: `{state.metrics.get('drc_available')}`",
            f"- Final board: `{state.board_path}`",
            "",
            "## Metrics",
            "```json",
            json.dumps(state.metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Artifacts",
            "- `final.kicad_pcb`: exported board when the run reaches `EXPORT_RESULT`.",
            "- `state.json`: complete replayable agent state.",
            "- `run_history.jsonl`: ordered tool calls and fallback decisions.",
            "- `round_*/candidate_ranking.json`: scored candidate list for each round.",
            "- `round_*/selected_candidate.json`: selected placement candidate for each round.",
            "",
            "## Notes",
            "- Real placement requires CUDA because the bundled `pcbplace.infer_layout` enforces CUDA.",
            "- Real Freerouting mode requires configured DSN export and SES import adapter commands.",
            "- If routing or DRC is unavailable and workflow fallback is enabled, the agent exports a placement-level artifact and records the fallback in `acceptance_reason`.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


