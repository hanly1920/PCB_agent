from __future__ import annotations

from pcbplace.infer import _resolve_inference_action_scoring
from pcbplace.policy_runtime import (
    DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    DEFAULT_ROLLOUT_REGION_ALPHA,
)
from pcbplace.train import _summarize_replay_rollout_episodes


def test_inference_action_scoring_defaults_to_checkpoint_metadata():
    objective_alpha, region_alpha, sources = _resolve_inference_action_scoring(
        {
            "action_scoring": {
                "version": 1,
                "objective_alpha": 2.5,
                "region_alpha": 0.75,
            }
        },
        requested_objective_alpha=None,
        requested_region_alpha=None,
    )

    assert objective_alpha == 2.5
    assert region_alpha == 0.75
    assert sources == {
        "objective_alpha": "checkpoint",
        "region_alpha": "checkpoint",
    }


def test_inference_action_scoring_explicit_override_wins_per_field():
    objective_alpha, region_alpha, sources = _resolve_inference_action_scoring(
        {
            "action_scoring": {
                "objective_alpha": 2.5,
                "region_alpha": 0.75,
            }
        },
        requested_objective_alpha=3.0,
        requested_region_alpha=None,
    )

    assert objective_alpha == 3.0
    assert region_alpha == 0.75
    assert sources == {
        "objective_alpha": "explicit_override",
        "region_alpha": "checkpoint",
    }


def test_inference_action_scoring_legacy_checkpoint_uses_runtime_defaults():
    objective_alpha, region_alpha, sources = _resolve_inference_action_scoring(
        {},
        requested_objective_alpha=None,
        requested_region_alpha=None,
    )

    assert objective_alpha == DEFAULT_ROLLOUT_OBJECTIVE_ALPHA
    assert region_alpha == DEFAULT_ROLLOUT_REGION_ALPHA
    assert sources == {
        "objective_alpha": "legacy_checkpoint_default",
        "region_alpha": "legacy_checkpoint_default",
    }


def test_replay_rollout_stats_include_first_step_no_legal_failures():
    stats = _summarize_replay_rollout_episodes(
        [
            {
                "complete": False,
                "failure_reason": "no_legal_action",
                "partial_obj": 100.0,
                "steps": 0,
            },
            {
                "complete": True,
                "failure_reason": None,
                "partial_obj": 40.0,
                "steps": 3,
            },
        ]
    )

    assert stats["complete_rate"] == 0.5
    assert stats["illegal_rate"] == 0.0
    assert stats["no_legal_rate"] == 0.5
    assert stats["raw_objective_mean"] == 70.0
