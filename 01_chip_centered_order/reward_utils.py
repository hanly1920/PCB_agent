"""Shared reward computation utilities aligning with the PCBAgent paper.

The PCBAgent paper (ASPDAC'25) defines the step reward during trajectory
rollout as the *decrease* in HPWL between two consecutive placement steps,
while penalising violations of soft constraints such as surface-layer wires.
We expose helpers that reproduce this behaviour for both offline dataset
processing and online policy rollouts so that rewards, returns-to-go and
conditioning tokens stay consistent everywhere.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

DEFAULT_SLW_PENALTY_WEIGHT = 1.0
DEFAULT_HPWL_THRESHOLD_WEIGHT = 1.0


def step_reward(
    prev_hpwl: float,
    curr_hpwl: float,
    prev_slw: float = 0.0,
    curr_slw: float = 0.0,
    *,
    hpwl_threshold: Optional[float] = None,
    hpwl_penalty_weight: float = DEFAULT_HPWL_THRESHOLD_WEIGHT,
    slw_penalty_weight: float = DEFAULT_SLW_PENALTY_WEIGHT,
    normalize: bool = True,
) -> float:
    """Compute the immediate reward for a single placement step.

    The reward mirrors Algorithm 1 from PCBAgent: an HPWL decrease yields
    positive reward, while increases in soft-constraint terms impose penalties.
    Optional penalties can be applied when HPWL exceeds a soft threshold.

    Args:
        prev_hpwl: HPWL before executing the current action.
        curr_hpwl: HPWL after executing the current action.
        prev_slw: SLW metric before the action.
        curr_slw: SLW metric after the action.
        hpwl_threshold: Soft HPWL limit; overflow reduces the reward.
        hpwl_penalty_weight: Weight applied to HPWL overflow.
        slw_penalty_weight: Weight applied to SLW differences.

    Returns:
        The scalar reward to attach to the current timestep.
    """
    hpwl_improvement = prev_hpwl - curr_hpwl
    if normalize:
        hpwl_scale = max(max(prev_hpwl, curr_hpwl), 1.0)
        hpwl_term = hpwl_improvement / hpwl_scale
    else:
        hpwl_term = hpwl_improvement

    reward = hpwl_term

    if hpwl_threshold is not None and curr_hpwl > hpwl_threshold:
        overflow = curr_hpwl - hpwl_threshold
        if normalize:
            overflow_scale = max(hpwl_threshold, 1.0)
            reward -= hpwl_penalty_weight * (overflow / overflow_scale)
        else:
            reward -= hpwl_penalty_weight * overflow

    slw_difference = prev_slw - curr_slw
    if normalize:
        slw_scale = max(abs(prev_slw), abs(curr_slw), 1.0)
        slw_term = slw_difference / slw_scale
    else:
        slw_term = slw_difference
    reward += slw_penalty_weight * slw_term

    return reward


def rewards_from_series(
    hpwl_series: Sequence[float],
    *,
    initial_hpwl: float = 0.0,
    slw_series: Optional[Sequence[float]] = None,
    initial_slw: float = 0.0,
    hpwl_threshold: Optional[float] = None,
    hpwl_penalty_weight: float = DEFAULT_HPWL_THRESHOLD_WEIGHT,
    slw_penalty_weight: float = DEFAULT_SLW_PENALTY_WEIGHT,
    normalize: bool = True,
) -> List[float]:
    """Generate per-step rewards given HPWL (and optional SLW) traces.

    ``hpwl_series`` should contain the cumulative HPWL after each placement
    action.  ``slw_series`` is optional; when omitted the soft-constraint term is
    ignored.
    """
    rewards: List[float] = []
    prev_hpwl = initial_hpwl
    prev_slw = initial_slw

    for idx, curr_hpwl in enumerate(hpwl_series):
        curr_slw = slw_series[idx] if slw_series is not None else prev_slw
        reward = step_reward(
            prev_hpwl,
            curr_hpwl,
            prev_slw,
            curr_slw,
            hpwl_threshold=hpwl_threshold,
            hpwl_penalty_weight=hpwl_penalty_weight,
            slw_penalty_weight=slw_penalty_weight,
            normalize=normalize,
        )
        rewards.append(reward)
        prev_hpwl = curr_hpwl
        prev_slw = curr_slw

    return rewards


def returns_from_rewards(rewards: Iterable[float]) -> List[float]:
    """Compute reverse cumulative sums (returns-to-go)."""
    returns: List[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running += reward
        returns.append(running)
    returns.reverse()
    return returns


def trajectory_score(lambda1: float, lambda2: float, hpwl_final: float, nslw_final: float) -> float:
    """Score definition from the PCBAgent paper (Equation 2)."""
    safe_hpwl = max(hpwl_final, 1e-6)
    return lambda1 * (1.0 / safe_hpwl) + lambda2 * nslw_final
