import math

from pcbplace.train import (
    INCOMPLETE_OBJECTIVE_PENALTY,
    _failed_rollout_objective_with_penalties,
    _normalize_replay_episode_score,
)


def test_failed_rollout_objective_uses_partial_plus_penalties_not_sentinel():
    final_obj, penalties = _failed_rollout_objective_with_penalties(
        partial_obj=120.0,
        placed_count=6,
        expected_count=10,
        failure_reason="illegal_action:overlap",
        terminated=True,
    )

    assert math.isfinite(final_obj)
    assert final_obj < INCOMPLETE_OBJECTIVE_PENALTY
    assert final_obj > 120.0
    assert penalties["missing_penalty"] > 0.0
    assert penalties["illegal_penalty"] > 0.0
    assert penalties["terminal_penalty"] > 0.0
    assert penalties["total_failure_penalty"] == (
        penalties["missing_penalty"]
        + penalties["illegal_penalty"]
        + penalties["terminal_penalty"]
    )


def test_failed_rollout_normalization_keeps_finite_relative_score():
    final_obj, penalties = _failed_rollout_objective_with_penalties(
        partial_obj=50.0,
        placed_count=5,
        expected_count=8,
        failure_reason="incomplete_layout",
        terminated=False,
    )
    episode = {
        "task_path": "board.json",
        "actions_flat": [1, 2, 3],
        "final_obj": final_obj,
        "partial_obj": 50.0,
        "failure_penalty": penalties["total_failure_penalty"],
        "complete": False,
        "score": -final_obj,
    }

    normalized = _normalize_replay_episode_score(episode, reference_obj=100.0)

    assert math.isfinite(normalized["score"])
    assert normalized["score"] > -INCOMPLETE_OBJECTIVE_PENALTY
    assert normalized["final_obj"] == final_obj
