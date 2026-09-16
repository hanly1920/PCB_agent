import json
from pathlib import Path

from pcb_agent.llm import DSLCompiler
from pcb_agent.config import LLMConfig
from pcb_agent.schemas import LayoutDSL, Constraint, LayoutCandidate
from pcb_agent.agent.orchestrator import AgentOrchestrator
from pcb_agent.tools.analyzer import LayoutAnalyzer
from pcb_agent.tools.dsl_adapter import DSLTaskAdapter
from pcb_agent.tools.repair import RepairPlanner


def _task(tmp_path: Path) -> Path:
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "board": {"bbox_mm": [0, 0, 40, 30], "grid_mm": 1.0},
                "components": [
                    {"ref": "J1", "size_mm": [4, 4], "fixed": False},
                    {"ref": "U1", "size_mm": [4, 4], "fixed": False},
                    {"ref": "C1", "size_mm": [2, 2], "fixed": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dsl_adapter_writes_edge_and_near_constraints(tmp_path):
    task = _task(tmp_path)
    dsl = DSLCompiler(LLMConfig(provider="mock")).compile("J1靠左，U1居中，C*靠近U1，间距0.2mm")

    patched, observations, metrics = DSLTaskAdapter.materialize_task(task, dsl)

    data = json.loads(patched.read_text(encoding="utf-8"))
    comps = {c["ref"]: c for c in data["components"]}
    assert not observations
    assert metrics["applied_constraint_count"] >= 3
    assert comps["J1"]["side_preference"] == "left"
    assert comps["J1"]["region_type"] == "edge_left"
    assert comps["U1"]["region_type"] == "core"
    assert comps["C1"]["anchor_ref"] == "U1"
    assert data["rules"]["agent_clearance_mm"] == 0.2


def test_analyzer_prefers_candidate_that_satisfies_edge_constraint(tmp_path):
    task = _task(tmp_path)
    dsl = LayoutDSL(
        hard_constraints=[Constraint(type="edge", refs=["J1"], side="left", priority="hard")],
        objective_weights={"user_preference": 5.0},
    )
    patched, _, _ = DSLTaskAdapter.materialize_task(task, dsl)
    bad = LayoutCandidate(
        candidate_id="bad",
        placements={"J1": {"x": 35, "y": 10}, "U1": {"x": 20, "y": 15}, "C1": {"x": 22, "y": 15}},
        objective=0.0,
        legal=True,
        complete=True,
    )
    good = LayoutCandidate(
        candidate_id="good",
        placements={"J1": {"x": 2, "y": 10}, "U1": {"x": 20, "y": 15}, "C1": {"x": 22, "y": 15}},
        objective=10.0,
        legal=True,
        complete=True,
    )

    result = LayoutAnalyzer().run([bad, good], patched, dsl)

    assert result.ok
    assert result.artifacts["best_candidate"]["candidate_id"] == "good"
    assert result.artifacts["ranking"][0]["hard_violation_count"] == 0
    assert result.artifacts["ranking"][1]["hard_violation_count"] == 1


def test_repair_planner_turns_violations_into_local_replay_plan():
    dsl = LayoutDSL()
    drc_report = {
        "violations": [
            {"category": "clearance", "refs": ["U1", "C1"], "severity": "error"},
        ]
    }
    metrics = {"best_hard_violations": [{"type": "edge", "ref": "J1"}]}

    result = RepairPlanner().run(drc_report, metrics, dsl)

    plan = result.artifacts["plan"]
    assert result.ok
    assert plan["action"] == "local_replacement"
    assert plan["freeze_other_refs"]
    assert plan["dsl_patch"]["movable_refs"] == ["C1", "J1", "U1"]
    assert plan["dsl_patch"]["objective_weights"]["spacing"] > 0.32


def test_replay_task_freezes_non_target_refs(tmp_path):
    task = _task(tmp_path)
    selected = {
        "placements": {
            "J1": {"x": 2, "y": 10, "rotation": 0},
            "U1": {"x": 20, "y": 15, "rotation": 0},
            "C1": {"x": 22, "y": 15, "rotation": 90},
        }
    }
    plan = {"freeze_other_refs": True, "target_refs": ["C1"], "action": "local_replacement"}

    replay = AgentOrchestrator._write_replay_task(task, selected, plan, tmp_path / "replay.json")

    data = json.loads(replay.read_text(encoding="utf-8"))
    comps = {c["ref"]: c for c in data["components"]}
    assert comps["J1"]["fixed_source"] == "agent_replay_freeze"
    assert comps["U1"]["fixed"] is True
    assert "fixed" not in comps["C1"]
