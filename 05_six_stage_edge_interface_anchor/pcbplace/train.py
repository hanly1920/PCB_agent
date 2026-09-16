from __future__ import annotations
import copy
import inspect
import math
import os
import random
import signal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from .dataset import task_from_json, normalize_sequence_policy
from .utils import load_json
from .env import PlacementEnv
from .model import MaskedPolicy, ModelConfig
from .region_prior import (
    REGION_TYPE_NAMES,
    SEMANTIC_CLASS_NAMES,
    SIDE_PREFERENCE_NAMES,
    SUBZONE_NAMES,
    PAIRWISE_RELATION_NAMES,
    RegionPriorConfig,
    load_region_targets_for_task,
    region_heatmap_num_bins,
)
from .env_cuda import (
    objective_delta_mask_cuda,
    action_features_cuda,
    action_prior_total_cuda,
    ACTION_CONDITIONED_ACTION_FEAT_DIM,
    build_context_tokens_cuda,
    action_mask_and_bias_cuda,
    action_region_prior_from_predictions_cuda,
)
from .policy_runtime import (
    flatten_action as _flatten_action,
    unflatten_action as _unflatten_action,
    cached_region_indices_cuda as _cached_region_indices_cuda,
    score_actions_batched,
    policy_log_probs_batched,
    ACTION_SCORING_VERSION,
    DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
    DEFAULT_ROLLOUT_REGION_ALPHA,
)

INCOMPLETE_OBJECTIVE_PENALTY = 1.0e30
FAILED_ROLLOUT_MIN_OBJECTIVE_SCALE = 1.0
FAILED_ROLLOUT_MISSING_COMPONENT_PENALTY_SCALE = 1.0
FAILED_ROLLOUT_ILLEGAL_PENALTY_SCALE = 1.0
FAILED_ROLLOUT_TERMINAL_PENALTY_SCALE = 0.25


ENV_CUDA_KWARG_DEFAULTS: Dict[str, float] = {
    'min_spacing_mm': 0.2,
    'alignment_bonus': 0.05,
    'edge_bonus': 0.15,
    'edge_eps_mm': 1.5,
    'non_interface_edge_penalty': 10.0,
    'non_interface_edge_margin_mm': 2.5,
    'density_penalty': 3.0,
    'density_radius_mm': 4.0,
    'interior_penalty': 1.0,
    'interior_margin_ratio': 0.18,
    'hpwl_weight': 1.0,
    'w_hpwl_weight': 0.20,
    'nslw_weight': 0.05,
    'region_weight': 0.55,
    'module_region_weight': 0.35,
    'module_floorplan_weight': 0.0,
    'module_region_bias': 0.20,
    'module_region_margin_mm': 2.0,
    'module_floorplan_separation_mm': 2.0,
    'module_floorplan_overlap_scale': 1.0,
    'module_floorplan_channel_scale': 0.65,
    'module_floorplan_compact_scale': 0.20,
    'module_floorplan_region_scale': 0.60,
    'conn_weight': 0.50,
    'objective_align_weight': 0.28,
    'group_weight': 0.12,
    'anchor_weight': 0.18,
    'boundary_group_weight': 0.22,
    'pitch_weight': 0.22,
    'orientation_weight': 0.14,
    'objective_edge_clearance_weight': 0.40,
    'objective_interior_weight': 0.30,
    'objective_density_weight': 0.45,
    'objective_soft_spacing_weight': 0.32,
    'objective_neatness_weight': 0.12,
    'edge_band_ratio': 0.12,
    'edge_band_center_ratio': 0.55,
    'soft_spacing_same_group_extra_mm': 0.6,
    'soft_spacing_cross_group_extra_mm': 1.4,
    'soft_spacing_large_extra_mm': 0.7,
    'same_group_density_scale': 0.40,
    'critical_neighbor_density_scale': 0.25,
    'anchor_group_density_scale': 0.50,
    'large_pair_density_scale': 1.20,
}

ENV_CUDA_KWARG_KEYS: Tuple[str, ...] = tuple(ENV_CUDA_KWARG_DEFAULTS.keys())


class _VisdomTrainLogger:
    """Optional Visdom line logger used by CUDA training.

    Import is lazy so normal training does not require visdom to be installed.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        server: str = "http://localhost",
        port: int = 8097,
        env: str = "pcb_autoplace",
        prefix: str = "train",
    ) -> None:
        self.enabled = bool(enabled)
        self.viz = None
        self.env = str(env)
        self.prefix = str(prefix)
        self.warned = False
        if not self.enabled:
            return
        try:
            from visdom import Visdom  # type: ignore
            self.viz = Visdom(server=server, port=int(port), env=self.env)
            if not self.viz.check_connection(timeout_seconds=2):
                print(f"[visdom] cannot connect to {server}:{int(port)} env={self.env}; disable Visdom logging.")
                self.enabled = False
                self.viz = None
            else:
                print(f"[visdom] enabled server={server}:{int(port)} env={self.env} prefix={self.prefix}")
        except Exception as exc:
            print(f"[visdom] disabled: {exc}")
            self.enabled = False
            self.viz = None

    def line(self, name: str, step: int, value: Any) -> None:
        if not self.enabled or self.viz is None:
            return
        try:
            if value is None:
                return
            y = float(value)
            if not math.isfinite(y):
                return
            win = f"{self.prefix}/{name}"
            self.viz.line(
                X=np.array([int(step)], dtype=np.float32),
                Y=np.array([y], dtype=np.float32),
                env=self.env,
                win=win,
                name=name,
                update="append",
                opts={"title": win, "xlabel": "step", "ylabel": name},
            )
        except Exception as exc:
            if not self.warned:
                print(f"[visdom] logging failed once; disabling Visdom logging: {exc}")
                self.warned = True
            self.enabled = False
            self.viz = None

    def scalars(self, step: int, values: Dict[str, Any]) -> None:
        for k, v in values.items():
            self.line(str(k), int(step), v)


# Canonical PlacementEnv kwarg -> accepted input names.
# The canonical names are what checkpoint env_config stores and what PlacementEnv
# receives.  The *_objective and reward/env aliases are the CLI names used by
# scripts/step3_train_masked.py and older checkpoints.
ENV_CUDA_KWARG_ALIASES: Dict[str, Tuple[str, ...]] = {
    'min_spacing_mm': ('min_spacing_mm',),
    'alignment_bonus': ('alignment_bonus', 'env_alignment_bonus'),
    'edge_bonus': ('edge_bonus', 'env_edge_bonus'),
    'edge_eps_mm': ('edge_eps_mm', 'env_edge_eps_mm'),
    'non_interface_edge_penalty': ('non_interface_edge_penalty', 'reward_non_interface_edge_penalty'),
    'non_interface_edge_margin_mm': ('non_interface_edge_margin_mm', 'reward_non_interface_edge_margin_mm'),
    'density_penalty': ('density_penalty', 'reward_density_penalty'),
    'density_radius_mm': ('density_radius_mm', 'reward_density_radius_mm'),
    'interior_penalty': ('interior_penalty', 'reward_interior_penalty'),
    'interior_margin_ratio': ('interior_margin_ratio', 'reward_interior_margin_ratio'),
    'hpwl_weight': ('hpwl_weight', 'objective_hpwl_weight'),
    'w_hpwl_weight': ('w_hpwl_weight', 'objective_w_hpwl_weight'),
    'nslw_weight': ('nslw_weight', 'objective_nslw_weight'),
    'region_weight': ('region_weight', 'objective_region_weight'),
    'module_region_weight': ('module_region_weight', 'objective_module_region_weight'),
    'module_floorplan_weight': ('module_floorplan_weight', 'objective_module_floorplan_weight'),
    'module_region_bias': ('module_region_bias',),
    'module_region_margin_mm': ('module_region_margin_mm',),
    'module_floorplan_separation_mm': ('module_floorplan_separation_mm',),
    'module_floorplan_overlap_scale': ('module_floorplan_overlap_scale',),
    'module_floorplan_channel_scale': ('module_floorplan_channel_scale',),
    'module_floorplan_compact_scale': ('module_floorplan_compact_scale',),
    'module_floorplan_region_scale': ('module_floorplan_region_scale',),
    'conn_weight': ('conn_weight', 'objective_conn_weight'),
    'objective_align_weight': ('objective_align_weight', 'align_weight'),
    'group_weight': ('group_weight', 'objective_group_weight'),
    'anchor_weight': ('anchor_weight', 'objective_anchor_weight'),
    'boundary_group_weight': ('boundary_group_weight', 'objective_boundary_group_weight'),
    'pitch_weight': ('pitch_weight', 'objective_pitch_weight'),
    'orientation_weight': ('orientation_weight', 'objective_orientation_weight'),
    'objective_edge_clearance_weight': ('objective_edge_clearance_weight', 'edge_clearance_weight'),
    'objective_interior_weight': ('objective_interior_weight', 'interior_weight'),
    'objective_density_weight': ('objective_density_weight', 'density_weight'),
    'objective_soft_spacing_weight': ('objective_soft_spacing_weight', 'soft_spacing_weight'),
    'objective_neatness_weight': ('objective_neatness_weight', 'neatness_weight'),
    'edge_band_ratio': ('edge_band_ratio',),
    'edge_band_center_ratio': ('edge_band_center_ratio',),
    'soft_spacing_same_group_extra_mm': ('soft_spacing_same_group_extra_mm',),
    'soft_spacing_cross_group_extra_mm': ('soft_spacing_cross_group_extra_mm',),
    'soft_spacing_large_extra_mm': ('soft_spacing_large_extra_mm',),
    'same_group_density_scale': ('same_group_density_scale',),
    'critical_neighbor_density_scale': ('critical_neighbor_density_scale',),
    'anchor_group_density_scale': ('anchor_group_density_scale',),
    'large_pair_density_scale': ('large_pair_density_scale',),
}


def validate_placement_env_kwargs(env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fail fast if train/infer env kwargs drift from PlacementEnv.

    CUDA kernels read their objective/mask knobs through the PlacementEnv object,
    so every CUDA-relevant knob must be present at all construction boundaries.
    """
    params = inspect.signature(PlacementEnv.__init__).parameters
    accepted = {
        name
        for name in params
        if name not in {'self', 'task'}
    }
    unknown = sorted(set(env_kwargs) - accepted)
    if unknown:
        raise ValueError(f'Unknown PlacementEnv kwargs: {unknown}')
    missing = [key for key in ENV_CUDA_KWARG_KEYS if key not in env_kwargs]
    if missing:
        raise ValueError(f'Missing PlacementEnv CUDA-training kwargs: {missing}')
    return {key: env_kwargs[key] for key in ENV_CUDA_KWARG_KEYS}


def _mapping_from_source(source: Any) -> Dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    if hasattr(source, '__dict__'):
        return dict(vars(source))
    raise TypeError(f'Unsupported env kwarg source type: {type(source)!r}')


def build_placement_env_kwargs(source: Any = None, **overrides: Any) -> Dict[str, Any]:
    """Single source of truth for PlacementEnv CUDA knobs.

    Accepts either a dict/argparse Namespace/checkpoint env_config in *source*,
    keyword overrides, or both.  It returns canonical PlacementEnv kwargs only.
    This lets step3 expert snapping, train(), checkpoint env_config, and infer()
    all use the exact same normalization and defaults.
    """
    src = _mapping_from_source(source)
    merged = dict(src)
    merged.update({k: v for k, v in overrides.items() if v is not None})

    env_kwargs: Dict[str, Any] = {}
    for canonical, default in ENV_CUDA_KWARG_DEFAULTS.items():
        value = None
        for key in ENV_CUDA_KWARG_ALIASES.get(canonical, (canonical,)):
            if key in merged and merged[key] is not None:
                value = merged[key]
                break
        if value is None:
            value = default
        env_kwargs[canonical] = float(value)

    return validate_placement_env_kwargs(env_kwargs)


DEFAULT_LEARNING_RATE = 3e-4

# Checkpoint section key -> train() argument name.  These are the configuration
# values that define continuation semantics and therefore must be restored
# before model/environment construction during a normal resume.
_RESUME_CONFIG_SECTION_MAP: Dict[str, Dict[str, str]] = {
    'k_schedule': {
        'kmax_final': 'kmax_final',
        'warmup': 'warmup',
        'steps': 'steps',
        'max_main_steps': 'max_main_steps',
        'adaptive': 'adaptive_k',
        'fixed_k_steps': 'fixed_k_steps',
        'curriculum_window_mode': 'curriculum_window_mode',
        'adaptive_drop_ratio': 'adaptive_k_drop_ratio',
        'adaptive_drop_ratio_first': 'adaptive_k_drop_ratio_first',
        'adaptive_probe_steps': 'adaptive_k_probe_steps',
        'adaptive_min_steps': 'adaptive_k_min_steps',
        'adaptive_ema_beta': 'adaptive_k_ema_beta',
        'adaptive_max_steps_per_k': 'adaptive_k_max_steps_per_k',
        'adaptive_max_visits_target_slack': 'adaptive_k_max_visits_target_slack',
        'adaptive_ready_ratio': 'adaptive_k_ready_ratio',
        'adaptive_outlier_release_factor': 'adaptive_k_outlier_release_factor',
    },
    'board_graduation': {
        'ema_beta': 'board_full_ema_beta',
        'probe_visits': 'board_full_probe_visits',
        'drop_ratio': 'board_full_drop_ratio',
        'min_visits': 'board_full_min_visits',
        'plateau_patience': 'board_full_plateau_patience',
        'plateau_rel_change': 'board_full_plateau_rel_change',
        'max_visits': 'board_full_max_visits',
    },
    'teacher': {
        'tau': 'teacher_tau',
        'lambda_region_prior': 'teacher_lambda_region_prior',
        'lambda_prior_region_heatmap': 'teacher_lambda_prior_region_heatmap',
        'lambda_conn_prior': 'teacher_lambda_conn_prior',
        'lambda_anchor_prior': 'teacher_lambda_anchor_prior',
        'lambda_module_prior': 'teacher_lambda_module_prior',
        'lambda_spacing_prior': 'teacher_lambda_spacing_prior',
        'lambda_edge_prior': 'teacher_lambda_edge_prior',
        'metric_weight': 'teacher_metric_weight',
        'topk': 'teacher_topk',
        'objective_delta_max': 'teacher_objective_delta_max',
        'gate_rollout': 'teacher_gate_rollout',
    },
    'region_prior': {
        'enabled': 'region_prior_enabled',
        'grid_x': 'region_grid_x',
        'grid_y': 'region_grid_y',
        'heatmap_sigma_cells': 'region_heatmap_sigma_cells',
        'legacy_zone_edge_ratio': 'legacy_region_zone_edge_ratio',
        'legacy_zone_core_ratio': 'legacy_region_zone_core_ratio',
        'heatmap_action_prior_weight': 'region_heatmap_action_prior_weight',
        'aux_heatmap_weight': 'region_aux_heatmap_weight',
        'aux_prior_consistency_weight': 'region_aux_prior_consistency_weight',
        'aux_semantic_weight': 'region_aux_semantic_weight',
        'aux_side_weight': 'region_aux_side_weight',
        'aux_subzone_weight': 'region_aux_subzone_weight',
        'aux_pairwise_weight': 'region_aux_pairwise_weight',
    },
    'geometry_distill': {
        'xy_weight': 'geometry_xy_weight',
        'rot_weight': 'geometry_rot_weight',
        'align_offset_weight': 'geometry_align_offset_weight',
        'boundary_axis_weight': 'geometry_boundary_axis_weight',
        # Removed unsafe expert-module-bbox distillation. The CLI argument is kept as a no-op for old configs.
        'module_bbox_weight': 'geometry_module_bbox_weight',
    },
    'expert_mix': {
        'start': 'expert_mix_start',
        'end': 'expert_mix_end',
        'anneal_steps': 'expert_mix_anneal_steps',
    },
    'replay': {
        'enabled': 'replay_finetune',
        'iters': 'replay_iters',
        'rollouts_per_iter': 'replay_rollouts_per_iter',
        'update_steps': 'replay_update_steps',
        'batch_size': 'replay_batch_size',
        'capacity': 'replay_capacity',
        'alpha': 'replay_alpha',
        'temp': 'replay_temp',
        'action_ce_coef': 'replay_action_ce_coef',
        'policy_gradient_coef': 'replay_policy_gradient_coef',
        'entropy_coef': 'replay_entropy_coef',
        'rollout_temperature': 'replay_rollout_temperature',
        'pg_policy_mode': 'replay_pg_policy_mode',
        'baseline_beta': 'replay_baseline_beta',
        'advantage_clip': 'replay_advantage_clip',
        'k': 'replay_k',
    },
}

# Canonical PlacementEnv checkpoint key -> train() argument name.
_RESUME_ENV_ARG_MAP: Dict[str, str] = {
    'min_spacing_mm': 'min_spacing_mm',
    'alignment_bonus': 'env_alignment_bonus',
    'edge_bonus': 'env_edge_bonus',
    'edge_eps_mm': 'env_edge_eps_mm',
    'non_interface_edge_penalty': 'reward_non_interface_edge_penalty',
    'non_interface_edge_margin_mm': 'reward_non_interface_edge_margin_mm',
    'density_penalty': 'reward_density_penalty',
    'density_radius_mm': 'reward_density_radius_mm',
    'interior_penalty': 'reward_interior_penalty',
    'interior_margin_ratio': 'reward_interior_margin_ratio',
    'hpwl_weight': 'objective_hpwl_weight',
    'w_hpwl_weight': 'objective_w_hpwl_weight',
    'nslw_weight': 'objective_nslw_weight',
    'region_weight': 'objective_region_weight',
    'module_region_weight': 'objective_module_region_weight',
    'module_floorplan_weight': 'objective_module_floorplan_weight',
    'module_region_bias': 'module_region_bias',
    'module_region_margin_mm': 'module_region_margin_mm',
    'module_floorplan_separation_mm': 'module_floorplan_separation_mm',
    'module_floorplan_overlap_scale': 'module_floorplan_overlap_scale',
    'module_floorplan_channel_scale': 'module_floorplan_channel_scale',
    'module_floorplan_compact_scale': 'module_floorplan_compact_scale',
    'module_floorplan_region_scale': 'module_floorplan_region_scale',
    'conn_weight': 'objective_conn_weight',
    'objective_align_weight': 'objective_align_weight',
    'group_weight': 'objective_group_weight',
    'anchor_weight': 'objective_anchor_weight',
    'boundary_group_weight': 'objective_boundary_group_weight',
    'pitch_weight': 'objective_pitch_weight',
    'orientation_weight': 'objective_orientation_weight',
    'objective_edge_clearance_weight': 'objective_edge_clearance_weight',
    'objective_interior_weight': 'objective_interior_weight',
    'objective_density_weight': 'objective_density_weight',
    'objective_soft_spacing_weight': 'objective_soft_spacing_weight',
    'objective_neatness_weight': 'objective_neatness_weight',
    'edge_band_ratio': 'edge_band_ratio',
    'edge_band_center_ratio': 'edge_band_center_ratio',
    'soft_spacing_same_group_extra_mm': 'soft_spacing_same_group_extra_mm',
    'soft_spacing_cross_group_extra_mm': 'soft_spacing_cross_group_extra_mm',
    'soft_spacing_large_extra_mm': 'soft_spacing_large_extra_mm',
    'same_group_density_scale': 'same_group_density_scale',
    'critical_neighbor_density_scale': 'critical_neighbor_density_scale',
    'anchor_group_density_scale': 'anchor_group_density_scale',
    'large_pair_density_scale': 'large_pair_density_scale',
}


def _resume_train_config_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten saved continuation configuration to train() argument names.

    Missing sections/keys are intentionally omitted so legacy checkpoints keep
    the caller's current value for fields they did not store.
    """
    restored: Dict[str, Any] = {}
    if checkpoint.get('max_tokens') is not None:
        restored['max_tokens'] = checkpoint['max_tokens']
    if checkpoint.get('sequence_policy') is not None:
        restored['sequence_policy'] = checkpoint['sequence_policy']

    for section_name, key_map in _RESUME_CONFIG_SECTION_MAP.items():
        section = checkpoint.get(section_name)
        if not isinstance(section, dict):
            continue
        for checkpoint_key, train_arg in key_map.items():
            if checkpoint_key in section:
                restored[train_arg] = section[checkpoint_key]

    env_section = checkpoint.get('env_config')
    if isinstance(env_section, dict):
        env_config = build_placement_env_kwargs(env_section)
        for canonical, train_arg in _RESUME_ENV_ARG_MAP.items():
            restored[train_arg] = env_config[canonical]
    return restored


def _validate_model_config(cfg: ModelConfig) -> ModelConfig:
    if int(cfg.d_model) <= 0:
        raise ValueError(f'model_d_model must be > 0; got {cfg.d_model}')
    if int(cfg.nhead) <= 0:
        raise ValueError(f'model_nhead must be > 0; got {cfg.nhead}')
    if int(cfg.num_layers) <= 0:
        raise ValueError(f'model_num_layers must be > 0; got {cfg.num_layers}')
    if int(cfg.d_model) % int(cfg.nhead) != 0:
        raise ValueError(
            f'model_d_model must be divisible by model_nhead; '
            f'got d_model={cfg.d_model}, nhead={cfg.nhead}'
        )
    if not (0.0 <= float(cfg.dropout) < 1.0):
        raise ValueError(f'model_dropout must be in [0, 1); got {cfg.dropout}')
    return cfg


def _model_config_from_checkpoint(
    checkpoint: Optional[Dict[str, Any]],
    fallback: Optional[ModelConfig] = None,
) -> ModelConfig:
    base = fallback or ModelConfig()
    if not checkpoint:
        return _validate_model_config(base)
    raw = checkpoint.get('model_cfg')
    if not isinstance(raw, dict):
        return _validate_model_config(base)
    allowed = set(ModelConfig.__dataclass_fields__.keys())
    values = dict(vars(base))
    values.update({key: raw[key] for key in allowed if key in raw})
    return _validate_model_config(ModelConfig(**values))


def _optimizer_learning_rates_from_checkpoint(checkpoint: Dict[str, Any]) -> List[float]:
    state = checkpoint.get('optimizer_state')
    if not isinstance(state, dict):
        return []
    groups = state.get('param_groups')
    if not isinstance(groups, list):
        return []
    rates: List[float] = []
    for group in groups:
        if isinstance(group, dict) and group.get('lr') is not None:
            rates.append(float(group['lr']))
    return rates


def _set_optimizer_learning_rate(opt: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in opt.param_groups:
        group['lr'] = float(learning_rate)


def _validate_resume_model_metadata(checkpoint: Dict[str, Any], region_cfg: RegionPriorConfig) -> None:
    """Fail early before strict state loading when architecture metadata drifts."""
    expected: Dict[str, int] = {
        'obs_dim': int(OBS_TOKEN_DIM),
        'action_feat_dim': int(ACTION_CONDITIONED_ACTION_FEAT_DIM),
        'num_region_heatmap_bins': int(region_heatmap_num_bins(region_cfg.grid_x, region_cfg.grid_y)),
        'num_semantic_classes': int(len(SEMANTIC_CLASS_NAMES)),
        'num_side_preferences': int(len(SIDE_PREFERENCE_NAMES)),
        'num_subzones': int(len(SUBZONE_NAMES)),
        'num_pairwise_relations': int(len(PAIRWISE_RELATION_NAMES)),
    }
    mismatches: List[str] = []
    for key, current in expected.items():
        if checkpoint.get(key) is None:
            continue
        saved = int(checkpoint[key])
        if saved != current:
            mismatches.append(f'{key}: checkpoint={saved}, current={current}')

    saved_shape = checkpoint.get('region_grid_shape')
    if isinstance(saved_shape, (list, tuple)) and len(saved_shape) == 2:
        current_shape = (int(region_cfg.grid_x), int(region_cfg.grid_y))
        checkpoint_shape = (int(saved_shape[0]), int(saved_shape[1]))
        if checkpoint_shape != current_shape:
            mismatches.append(
                f'region_grid_shape: checkpoint={checkpoint_shape}, current={current_shape}'
            )

    if mismatches:
        raise ValueError(
            'Resume checkpoint architecture is incompatible with the current training code: '
            + '; '.join(mismatches)
        )



def schedule_kmax(step: int, total_steps: int, hold_steps: int, kmax_final: int, power: float = 1.8) -> int:
    """
    Full-masking friendly k schedule:
    - Hold k=1 for `hold_steps`
    - Then cosine ramp to kmax_final by `total_steps`
    power>1 makes early growth slower (lower variance), later faster.
    """
    if kmax_final <= 1:
        return 1
    hold_steps = max(0, int(hold_steps))
    total_steps = max(1, int(total_steps))

    if step <= hold_steps:
        return 1

    denom = max(1, total_steps - hold_steps)
    u = min(1.0, float(step - hold_steps) / float(denom))           # 0..1
    s = 0.5 * (1.0 - math.cos(math.pi * u))                         # smooth 0..1
    s = s ** float(power)                                           # slower early, faster late

    k = 1.0 + (float(kmax_final) - 1.0) * s
    return max(1, min(kmax_final, int(round(k))))


def schedule_mix(step: int, total_steps: int, hold_steps: int, start: float, end: float, power: float = 1.6) -> float:
    """
    Cosine + hold schedule for alpha_expert:
    - Hold at `start` for `hold_steps`
    - Then cosine decay to `end` by `total_steps`
    """
    total_steps = max(1, int(total_steps))
    hold_steps = max(0, int(hold_steps))

    if total_steps <= hold_steps + 1:
        return float(end)
    if step <= hold_steps:
        return float(start)

    denom = max(1, total_steps - hold_steps)
    u = min(1.0, float(step - hold_steps) / float(denom))
    s = 0.5 * (1.0 - math.cos(math.pi * u))
    s = s ** float(power)
    return float(start + (end - start) * s)


_TYPES = ["interface", "mechanical", "chip", "capacitor", "resistor", "misc"]

# Must mirror env_cuda._CONTEXT_MODULE_TYPES and
# env_cuda._CONTEXT_MODULE_TOKEN_EXTRA_DIM, which build the runtime tokens.
MODULE_TOKEN_EXTRA_DIM = 8 + 5
OBS_TOKEN_DIM = len(_TYPES) + 10 + MODULE_TOKEN_EXTRA_DIM  # original 16 + module features = 29


@dataclass
class TeacherConfig:
    """Teacher distribution q(a) built from residual ΔJ(a) plus action-conditioned priors."""
    tau: float = 0.5
    lambda_region_prior: float = 0.15  # predicted heatmap action prior
    lambda_prior_region_heatmap: float = 0.10  # external inference-safe prior heatmap action prior
    lambda_conn_prior: float = 0.15
    lambda_anchor_prior: float = 0.20
    lambda_module_prior: float = 0.15
    lambda_spacing_prior: float = 0.08
    lambda_edge_prior: float = 0.05
    metric_weight: float = 0.15
    topk: int = 256
    objective_delta_max: Optional[float] = None
    gate_rollout: bool = True




def _soft_ce_from_distribution(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Soft-label cross entropy; equivalent to KL up to the target entropy constant."""
    if target_probs.dim() == 1:
        target_probs = target_probs.view(1, -1)
    if logits.dim() == 1:
        logits = logits.view(1, -1)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@dataclass(frozen=True)
class RegionSupervisionWeights:
    """Independent trust weights for region-related supervision streams.

    ``semantic`` is derived from semantic_review metadata. Expert placement
    heatmaps, teacher distillation, metric shaping, and runtime-prior
    consistency must not be implicitly downweighted just because a semantic
    label still needs review.
    """

    expert: float = 1.0
    semantic: float = 1.0
    teacher: float = 1.0
    metric: float = 1.0
    prior: float = 1.0


def _clamp01_float(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    return float(max(0.0, min(1.0, v)))


def _region_supervision_weights_from_row(row: Dict[str, Any]) -> RegionSupervisionWeights:
    """Build split supervision weights from a region-target row.

    Backward compatibility: legacy ``review_weight`` maps only to semantic
    supervision. All other streams default to full trust unless their own
    explicit weight is present.
    """

    semantic_default = row.get('semantic_supervision_weight', row.get('review_weight', 1.0))
    return RegionSupervisionWeights(
        expert=_clamp01_float(row.get('expert_supervision_weight', row.get('expert_weight', 1.0)), 1.0),
        semantic=_clamp01_float(semantic_default, 1.0),
        teacher=_clamp01_float(row.get('teacher_supervision_weight', row.get('teacher_weight', 1.0)), 1.0),
        metric=_clamp01_float(row.get('metric_supervision_weight', row.get('metric_weight', 1.0)), 1.0),
        prior=_clamp01_float(row.get('prior_supervision_weight', row.get('prior_consistency_review_weight', 1.0)), 1.0),
    )


def _cached_region_target_tensors(env: PlacementEnv, ref: str, task_path: str, region_cfg: RegionPriorConfig, device: torch.device):
    """Return expert heatmap + semantic aux targets for one current ref.

    expert_region_heatmap is train-only supervision. prior_region_heatmap is
    inference-safe input and is returned for a low-weight consistency regularizer.

    The hot training path clones PlacementEnv objects frequently.  Cache target
    payloads and GPU tensors on the shared _CachedBoardRuntime item when present
    so clones do not repeatedly parse JSON or re-upload identical tensors.
    """
    device_key = _device_cache_key(device) if '_device_cache_key' in globals() else str(device)
    cfg_key = (
        int(region_cfg.grid_x),
        int(region_cfg.grid_y),
        float(region_cfg.heatmap_sigma_cells),
        float(region_cfg.legacy_zone_edge_ratio),
        float(region_cfg.legacy_zone_core_ratio),
    )
    key = (
        str(ref),
        str(task_path),
        int(region_cfg.grid_x),
        int(region_cfg.grid_y),
        float(region_cfg.heatmap_sigma_cells),
        float(region_cfg.legacy_zone_edge_ratio),
        float(region_cfg.legacy_zone_core_ratio),
        device_key,
    )

    runtime_item = _runtime_item_from_env(env) if '_runtime_item_from_env' in globals() else None
    if runtime_item is not None:
        tensor_cache = getattr(runtime_item, 'region_target_tensors_by_key', None)
        if isinstance(tensor_cache, dict) and key in tensor_cache:
            return tensor_cache[key]
    else:
        cache = getattr(env, '_region_target_tensor_cache', None)
        if isinstance(cache, dict) and key in cache:
            return cache[key]

    if runtime_item is not None:
        payload_cache = getattr(runtime_item, 'region_targets_by_cfg', None)
        if isinstance(payload_cache, dict) and cfg_key in payload_cache:
            payload = payload_cache[cfg_key]
        else:
            payload = load_region_targets_for_task(
                str(task_path),
                int(region_cfg.grid_x),
                int(region_cfg.grid_y),
                float(region_cfg.heatmap_sigma_cells),
                float(region_cfg.legacy_zone_edge_ratio),
                float(region_cfg.legacy_zone_core_ratio),
            )
            if isinstance(payload_cache, dict):
                payload_cache[cfg_key] = payload
    else:
        payload = load_region_targets_for_task(
            str(task_path),
            int(region_cfg.grid_x),
            int(region_cfg.grid_y),
            float(region_cfg.heatmap_sigma_cells),
            float(region_cfg.legacy_zone_edge_ratio),
            float(region_cfg.legacy_zone_core_ratio),
        )

    row = payload.get(str(ref))
    if row is None or row.get('expert_region_heatmap') is None:
        out = None
    else:
        heat = np.asarray(row.get('expert_region_heatmap'), dtype=np.float32).reshape(-1)
        expected = int(region_heatmap_num_bins(region_cfg.grid_x, region_cfg.grid_y))
        if heat.size != expected:
            out = None
        else:
            if float(heat.sum()) <= 0.0:
                heat[:] = 1.0 / float(max(1, heat.size))
            else:
                heat = heat / float(heat.sum())
            prior_heat = row.get('prior_region_heatmap')
            if prior_heat is None:
                prior = np.full_like(heat, 1.0 / float(max(1, heat.size)), dtype=np.float32)
                prior_available = 0.0
            else:
                prior = np.asarray(prior_heat, dtype=np.float32).reshape(-1)
                if prior.size != expected or float(prior.sum()) <= 0.0:
                    prior = np.full_like(heat, 1.0 / float(max(1, heat.size)), dtype=np.float32)
                    prior_available = 0.0
                else:
                    prior = prior / float(prior.sum())
                    prior_available = 1.0
            heat_t = torch.as_tensor(heat, device=device, dtype=torch.float32).view(1, -1)
            prior_t = torch.as_tensor(prior, device=device, dtype=torch.float32).view(1, -1)
            semantic_t = torch.tensor([int(row.get('semantic_class', 0))], device=device, dtype=torch.long)
            side_t = torch.tensor([int(row.get('side_preference', 0))], device=device, dtype=torch.long)
            subzone_t = torch.tensor([int(row.get('subzone', 0))], device=device, dtype=torch.long)
            supervision_weights = _region_supervision_weights_from_row(row)
            prior_confidence = float(env._module_region_confidence_for_ref(ref)) if hasattr(env, '_module_region_confidence_for_ref') else 0.0
            prior_weight = prior_available * float(max(0.0, min(1.0, prior_confidence)))
            out = (heat_t, prior_t, semantic_t, side_t, subzone_t, supervision_weights, prior_weight)

    if runtime_item is not None:
        tensor_cache = getattr(runtime_item, 'region_target_tensors_by_key', None)
        if isinstance(tensor_cache, dict):
            tensor_cache[key] = out
    else:
        cache = getattr(env, '_region_target_tensor_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(env, '_region_target_tensor_cache', cache)
        cache[key] = out
    return out


def _validate_region_heatmap_grid_for_tasks(train_tasks: List[Dict[str, Any]], region_cfg: RegionPriorConfig) -> None:
    """Hard-fail when structured JSON heatmap grids drift from training args."""
    expected = (int(region_cfg.grid_x), int(region_cfg.grid_y))
    mismatches: List[str] = []
    missing: List[str] = []
    for item in train_tasks:
        path = str(item.get('path', ''))
        if not path:
            continue
        try:
            data = load_json(path)
        except Exception as exc:
            raise ValueError(f'Cannot read task JSON for heatmap grid check: {path}: {exc}') from exc
        meta_cfg = ((data.get('meta') or {}).get('region_heatmap_config') or {}) if isinstance(data, dict) else {}
        graph_cfg = (((data.get('graph') or {}).get('module_region_policy') or {}).get('region_heatmap_config') or {}) if isinstance(data, dict) else {}
        cfg = meta_cfg if meta_cfg else graph_cfg
        found = None
        if isinstance(cfg, dict) and cfg.get('grid_x') is not None and cfg.get('grid_y') is not None:
            try:
                found = (int(cfg.get('grid_x')), int(cfg.get('grid_y')))
            except Exception:
                found = None
        if found is None:
            # Fallback to the first prior/expert heatmap metadata if older files lack meta.
            for comp in data.get('components', []) if isinstance(data, dict) else []:
                if not isinstance(comp, dict):
                    continue
                payload = None
                prior = comp.get('prior') if isinstance(comp.get('prior'), dict) else {}
                if isinstance(prior.get('region_heatmap'), dict):
                    payload = prior.get('region_heatmap')
                elif isinstance((comp.get('expert') or {}).get('region_heatmap'), dict):
                    payload = (comp.get('expert') or {}).get('region_heatmap')
                if isinstance(payload, dict) and payload.get('grid_x') is not None and payload.get('grid_y') is not None:
                    found = (int(payload.get('grid_x')), int(payload.get('grid_y')))
                    break
        if found is None:
            missing.append(path)
        elif found != expected:
            mismatches.append(f'{path}: json_grid={found[0]}x{found[1]} train_grid={expected[0]}x{expected[1]}')
    if mismatches:
        preview = '\n'.join(mismatches[:10])
        extra = '' if len(mismatches) <= 10 else f'\n... and {len(mismatches) - 10} more'
        raise ValueError('Region heatmap grid mismatch. Regenerate structured JSON with the same --region_grid_x/--region_grid_y used for training:\n' + preview + extra)
    if missing:
        preview = ', '.join(missing[:5])
        extra = '' if len(missing) <= 5 else f', ... and {len(missing) - 5} more'
        print(f'[warn] {len(missing)} task(s) do not declare region_heatmap_config; continuing with train grid {expected[0]}x{expected[1]}: {preview}{extra}')


@dataclass
class GeometryDistillConfig:
    xy_weight: float = 0.05
    rot_weight: float = 0.02
    align_offset_weight: float = 0.03
    boundary_axis_weight: float = 0.03
    module_bbox_weight: float = 0.0


def _ref_expected_norm_from_probs(
    probs: torch.Tensor,
    feat_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_exp = (probs * feat_t[:, 0]).sum()
    y_exp = (probs * feat_t[:, 1]).sum()
    sin_exp = (probs * feat_t[:, 2]).sum()
    cos_exp = (probs * feat_t[:, 3]).sum()
    return x_exp, y_exp, sin_exp, cos_exp


def _align_axis_target(
    env: PlacementEnv,
    ref: str,
) -> Optional[Tuple[str, float]]:
    group = getattr(env, "_align_groups", {}).get(ref)
    if not group:
        return None
    peers = [pref for pref in getattr(env, "placed_order", []) if pref != ref and getattr(env, "_align_groups", {}).get(pref) == group]
    if not peers:
        return None
    xs = [float(env.placed[pref][0]) for pref in peers]
    ys = [float(env.placed[pref][1]) for pref in peers]
    if len(xs) >= 2 and len(ys) >= 2:
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        axis = "x" if spread_x <= spread_y else "y"
    else:
        axis = "x" if len(xs) >= len(ys) else "y"
    target = float(sum(xs) / max(1, len(xs))) if axis == "x" else float(sum(ys) / max(1, len(ys)))
    return axis, target


def _boundary_axis_target(
    env: PlacementEnv,
    ref: str,
) -> Optional[Tuple[str, str]]:
    group = getattr(env, "_same_side_groups", {}).get(ref)
    if not group:
        return None
    peers = [pref for pref in getattr(env, "placed_order", []) if pref != ref and getattr(env, "_same_side_groups", {}).get(pref) == group]
    if not peers:
        return None
    side = getattr(env, "_side_preferences", {}).get(ref, "free")
    if side not in {"edge_left", "edge_right", "edge_top", "edge_bottom"}:
        side = getattr(env, "_region_targets", {}).get(ref, "free")
    if side in {"edge_left", "edge_right"}:
        return "y", side
    if side in {"edge_top", "edge_bottom"}:
        return "x", side
    return None




def _geometry_distill_loss(
    *,
    env: PlacementEnv,
    ref: str,
    logits: torch.Tensor,
    feat_t: torch.Tensor,
    expert_action: Tuple[int, int, int],
    geom_cfg: GeometryDistillConfig,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    x_exp, y_exp, sin_exp, cos_exp = _ref_expected_norm_from_probs(probs, feat_t)

    w, h = env.grid_shape()
    R = len(env.rotations)
    gt_a = _flatten_action(int(expert_action[0]), int(expert_action[1]), int(expert_action[2]), w, h)
    tgt = feat_t[gt_a]

    loss = torch.tensor(0.0, device=logits.device)

    pred_xy = torch.stack([x_exp, y_exp], dim=0)
    tgt_xy = tgt[:2]
    loss = loss + float(geom_cfg.xy_weight) * F.smooth_l1_loss(pred_xy, tgt_xy, reduction="sum")

    pred_rot = torch.stack([sin_exp, cos_exp], dim=0)
    tgt_rot = tgt[2:4]
    loss = loss + float(geom_cfg.rot_weight) * F.smooth_l1_loss(pred_rot, tgt_rot, reduction="sum")

    xmin, ymin, xmax, ymax = env.task.bbox_mm
    bw = max(1e-6, float(xmax - xmin))
    bh = max(1e-6, float(ymax - ymin))

    align_target = _align_axis_target(env, ref)
    if align_target is not None:
        axis, axis_value = align_target
        if axis == "x":
            pred_off = x_exp - (float(axis_value) - float(xmin)) / bw
            tgt_off = tgt[0] - (float(axis_value) - float(xmin)) / bw
        else:
            pred_off = y_exp - (float(axis_value) - float(ymin)) / bh
            tgt_off = tgt[1] - (float(axis_value) - float(ymin)) / bh
        loss = loss + float(geom_cfg.align_offset_weight) * F.smooth_l1_loss(
            pred_off.view(1), tgt_off.view(1), reduction="sum"
        )

    boundary_target = _boundary_axis_target(env, ref)
    if boundary_target is not None:
        axis, _side = boundary_target
        pred_axis = x_exp if axis == "x" else y_exp
        tgt_axis = tgt[0] if axis == "x" else tgt[1]
        loss = loss + float(geom_cfg.boundary_axis_weight) * F.smooth_l1_loss(
            pred_axis.view(1), tgt_axis.view(1), reduction="sum"
        )


    return loss



def _smooth_l1_sum_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Row-wise smooth-L1 sum used by the batched geometry distillation loss."""
    return F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1)


def _geometry_distill_loss_batch(
    *,
    envs: List[PlacementEnv],
    refs: List[str],
    logits_batch: torch.Tensor,
    action_feat_batch: torch.Tensor,
    geometry_gt_indices: List[Optional[int]],
    geom_cfg: GeometryDistillConfig,
) -> torch.Tensor:
    """Batched variant of _geometry_distill_loss().

    The old suffix/replay hot paths executed one softmax and several tiny
    smooth_l1 kernels per row.  Here the expected x/y/rotation features and the
    mandatory xy/rot losses are computed in one batched GPU pass.  The optional
    alignment/boundary targets still require cheap Python metadata checks, but
    their loss math is also fused into row-wise tensor operations.
    """
    B = int(logits_batch.shape[0])
    if B <= 0:
        return torch.zeros((0,), device=logits_batch.device, dtype=logits_batch.dtype)

    zero = logits_batch.sum(dim=1) * 0.0
    valid_rows: List[int] = []
    safe_indices: List[int] = []
    for row, gt_idx in enumerate(geometry_gt_indices):
        if gt_idx is None:
            continue
        gi = int(gt_idx)
        if gi < 0 or gi >= int(action_feat_batch.shape[1]):
            continue
        valid_rows.append(row)
        safe_indices.append(gi)

    if not valid_rows:
        return zero

    row_idx = torch.tensor(valid_rows, device=logits_batch.device, dtype=torch.long)
    gt_idx_t = torch.tensor(safe_indices, device=logits_batch.device, dtype=torch.long)

    logits_sel = logits_batch.index_select(0, row_idx)
    feat_sel = action_feat_batch.index_select(0, row_idx)
    probs = torch.softmax(logits_sel, dim=-1)

    x_exp = (probs * feat_sel[:, :, 0]).sum(dim=-1)
    y_exp = (probs * feat_sel[:, :, 1]).sum(dim=-1)
    sin_exp = (probs * feat_sel[:, :, 2]).sum(dim=-1)
    cos_exp = (probs * feat_sel[:, :, 3]).sum(dim=-1)
    tgt = feat_sel[torch.arange(len(valid_rows), device=logits_batch.device), gt_idx_t]

    loss_valid = torch.zeros((len(valid_rows),), device=logits_batch.device, dtype=logits_batch.dtype)
    pred_xy = torch.stack([x_exp, y_exp], dim=-1)
    loss_valid = loss_valid + float(geom_cfg.xy_weight) * _smooth_l1_sum_rows(pred_xy, tgt[:, :2])
    pred_rot = torch.stack([sin_exp, cos_exp], dim=-1)
    loss_valid = loss_valid + float(geom_cfg.rot_weight) * _smooth_l1_sum_rows(pred_rot, tgt[:, 2:4])

    align_rows: List[int] = []
    align_pred_vals: List[torch.Tensor] = []
    align_tgt_vals: List[torch.Tensor] = []
    boundary_rows: List[int] = []
    boundary_pred_vals: List[torch.Tensor] = []
    boundary_tgt_vals: List[torch.Tensor] = []

    for local_pos, row in enumerate(valid_rows):
        env = envs[row]
        ref = refs[row]
        xmin, ymin, xmax, ymax = env.task.bbox_mm
        bw = max(1e-6, float(xmax - xmin))
        bh = max(1e-6, float(ymax - ymin))

        align_target = _align_axis_target(env, ref)
        if align_target is not None:
            axis, axis_value = align_target
            if axis == "x":
                offset = (float(axis_value) - float(xmin)) / bw
                align_pred_vals.append(x_exp[local_pos] - offset)
                align_tgt_vals.append(tgt[local_pos, 0] - offset)
            else:
                offset = (float(axis_value) - float(ymin)) / bh
                align_pred_vals.append(y_exp[local_pos] - offset)
                align_tgt_vals.append(tgt[local_pos, 1] - offset)
            align_rows.append(local_pos)

        boundary_target = _boundary_axis_target(env, ref)
        if boundary_target is not None:
            axis, _side = boundary_target
            if axis == "x":
                boundary_pred_vals.append(x_exp[local_pos])
                boundary_tgt_vals.append(tgt[local_pos, 0])
            else:
                boundary_pred_vals.append(y_exp[local_pos])
                boundary_tgt_vals.append(tgt[local_pos, 1])
            boundary_rows.append(local_pos)

    if align_rows:
        align_idx = torch.tensor(align_rows, device=logits_batch.device, dtype=torch.long)
        pred = torch.stack(align_pred_vals, dim=0).view(-1, 1)
        target = torch.stack(align_tgt_vals, dim=0).view(-1, 1)
        loss_valid = loss_valid.index_add(
            0,
            align_idx,
            float(geom_cfg.align_offset_weight) * _smooth_l1_sum_rows(pred, target),
        )

    if boundary_rows:
        boundary_idx = torch.tensor(boundary_rows, device=logits_batch.device, dtype=torch.long)
        pred = torch.stack(boundary_pred_vals, dim=0).view(-1, 1)
        target = torch.stack(boundary_tgt_vals, dim=0).view(-1, 1)
        loss_valid = loss_valid.index_add(
            0,
            boundary_idx,
            float(geom_cfg.boundary_axis_weight) * _smooth_l1_sum_rows(pred, target),
        )

    return zero.index_copy(0, row_idx, loss_valid)


def _gated_pairwise_relation_target(env: PlacementEnv, ref: str, placed: List[str], device: torch.device) -> torch.Tensor:
    """Build one gated pairwise target instead of multi-hot additive labels.

    Output order follows PAIRWISE_RELATION_NAMES: same_group, anchor_neighbor,
    critical_neighbor.  Priority is critical > direct_anchor > explicit_functional.
    Module fallback groups and generic connectivity are intentionally not marked
    as strong pairwise aux labels.
    """
    target = torch.zeros((1, len(PAIRWISE_RELATION_NAMES)), device=device)
    best = None
    for p in placed:
        rel_fn = getattr(env, '_pair_primary_relation', None)
        rel = rel_fn(ref, p) if rel_fn is not None else 'none'
        if rel == 'critical':
            best = 'critical'
            break
        if rel == 'anchor' and best not in {'critical'}:
            best = 'anchor'
        elif rel == 'functional_group' and best is None:
            best = 'functional_group'
    if best == 'functional_group':
        target[0, 0] = 1.0
    elif best == 'anchor':
        target[0, 1] = 1.0
    elif best == 'critical':
        target[0, 2] = 1.0
    return target

def _region_aux_losses(
    *,
    ref: str,
    task_path: str,
    env: PlacementEnv,
    region_cfg: RegionPriorConfig,
    region_heatmap_logits: Optional[torch.Tensor],
    semantic_class_logits: Optional[torch.Tensor],
    side_preference_logits: Optional[torch.Tensor],
    subzone_logits: Optional[torch.Tensor],
    pairwise_logits: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RegionSupervisionWeights]:
    zero = torch.tensor(0.0, device=device)
    if (not bool(region_cfg.enabled)) or region_heatmap_logits is None or semantic_class_logits is None:
        return zero, zero, zero, zero, zero, zero, RegionSupervisionWeights()
    target_tensors = _cached_region_target_tensors(env, ref, task_path, region_cfg, device)
    if target_tensors is None:
        return zero, zero, zero, zero, zero, zero, RegionSupervisionWeights()

    region_t, prior_region_t, semantic_t, side_t, subzone_t, supervision_weights, prior_weight = target_tensors
    loss_region_heatmap = _soft_ce_from_distribution(region_heatmap_logits.view(1, -1), region_t)
    loss_prior_consistency = float(prior_weight) * _soft_ce_from_distribution(region_heatmap_logits.view(1, -1), prior_region_t)
    loss_semantic = F.cross_entropy(semantic_class_logits.view(1, -1), semantic_t)
    loss_side = zero if side_preference_logits is None else F.cross_entropy(side_preference_logits.view(1, -1), side_t)
    loss_subzone = zero if subzone_logits is None else F.cross_entropy(subzone_logits.view(1, -1), subzone_t)

    relation_target = _gated_pairwise_relation_target(env, ref, list(env.placed_order), device)
    loss_pairwise = zero if pairwise_logits is None else F.binary_cross_entropy_with_logits(pairwise_logits.view(1, -1), relation_target)
    return loss_semantic, loss_region_heatmap, loss_prior_consistency, loss_side, loss_subzone, loss_pairwise, supervision_weights



# -------------------------
# Batched GPU Training (multiple boards at once)
# -------------------------

def _pad_1d_tensors(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad a list of [A_i] tensors into [B, Amax] on the same device."""
    if not tensors:
        raise ValueError("cannot pad an empty tensor list")
    lengths = [int(t.numel()) for t in tensors]
    max_len = max(lengths)
    if min(lengths) == max_len:
        return torch.stack([t.reshape(-1) for t in tensors], dim=0)
    out = torch.full((len(tensors), max_len), float(pad_value), device=tensors[0].device, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        n = int(t.numel())
        if n:
            out[i, :n] = t.reshape(-1)
    return out


def _pad_2d_tensors(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad a list of [A_i, F] tensors into [B, Amax, F]."""
    if not tensors:
        raise ValueError("cannot pad an empty tensor list")
    shapes = [tuple(t.shape) for t in tensors]
    if all(shape == shapes[0] for shape in shapes):
        return torch.stack(tensors, dim=0)
    max_len = max(int(t.shape[0]) for t in tensors)
    feat_dim = int(tensors[0].shape[1])
    out = torch.full((len(tensors), max_len, feat_dim), float(pad_value), device=tensors[0].device, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        n = int(t.shape[0])
        if n:
            out[i, :n, :] = t
    return out


def _left_pad_context_tokens(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length context tokens, keeping current-ref token at index -1."""
    if not tensors:
        raise ValueError("cannot pad an empty token list")
    lengths = [int(t.shape[0]) for t in tensors]
    max_len = max(lengths)
    feat_dim = int(tensors[0].shape[1])
    device = tensors[0].device
    dtype = tensors[0].dtype
    if min(lengths) == max_len and all(int(t.shape[1]) == feat_dim for t in tensors):
        out = torch.stack(tensors, dim=0)
        key_padding_mask = torch.zeros((len(tensors), max_len), device=device, dtype=torch.bool)
        return out, key_padding_mask
    out = torch.full((len(tensors), max_len, feat_dim), float(pad_value), device=device, dtype=dtype)
    key_padding_mask = torch.ones((len(tensors), max_len), device=device, dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = int(t.shape[0])
        if n <= 0:
            continue
        out[i, max_len - n:, :] = t
        key_padding_mask[i, max_len - n:] = False
    return out, key_padding_mask


@torch.no_grad()
def _build_teacher_distribution_batch_cuda(
    mask_batch: torch.Tensor,
    objective_batch: torch.Tensor,
    cfg: TeacherConfig,
    *,
    device: torch.device,
    region_batch: Optional[torch.Tensor] = None,
    action_prior_batch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build batched CUDA teacher distributions for padded action rows.

    Inputs:
      mask_batch/objective_batch: [B, Amax]
      region_batch: optional [B, Amax]
    Returns:
      q_batch:    [B, Amax]
      cand_batch: [B, Amax] bool
    """
    energy = objective_batch.clone()
    if region_batch is not None:
        energy = energy - float(cfg.lambda_region_prior) * region_batch
    if action_prior_batch is not None:
        energy = energy - action_prior_batch.to(device=device, dtype=energy.dtype)

    legal = mask_batch > 0.5
    cand = legal.clone()

    if cfg.objective_delta_max is not None:
        cand2 = cand & (objective_batch <= float(cfg.objective_delta_max))
        # Keep the original legal candidates for rows where the threshold removes everything.
        has_any = cand2.any(dim=1, keepdim=True)
        cand = torch.where(has_any, cand2, cand)

    topk = int(cfg.topk) if cfg.topk is not None else 0
    if topk > 0 and cand.shape[1] > topk:
        counts = cand.sum(dim=1)
        energy_for_topk = energy.masked_fill(~cand, float("inf"))
        _, top_idx = torch.topk(energy_for_topk, k=topk, dim=1, largest=False)
        cand_top = torch.zeros_like(cand)
        cand_top.scatter_(1, top_idx, True)
        cand_top = cand_top & cand
        cand = torch.where((counts > topk).unsqueeze(1), cand_top, cand)

    tlog = -energy / max(1e-6, float(cfg.tau))
    tlog2 = torch.full_like(tlog, -1e9)
    tlog2 = torch.where(cand, tlog, tlog2)
    q = torch.softmax(tlog2, dim=-1)
    return q, cand


def _policy_outputs_with_region_batch(
    model: MaskedPolicy,
    envs: List[PlacementEnv],
    refs: List[str],
    tokens_batch: torch.Tensor,
    action_feat_batch: torch.Tensor,
    region_cfg: RegionPriorConfig,
    token_key_padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """One batched model forward for multiple boards.

    tokens_batch:      [B, L, D]
    action_feat_batch: [B, Amax, F]
    returns logits:    [B, Amax]
    """
    enc = model.encode(tokens_batch, key_padding_mask=token_key_padding_mask)
    q_last = model.q_proj(enc[:, -1, :])
    logits = model.action_logits_from_query_batched(q_last, action_feat_batch)

    if not bool(region_cfg.enabled):
        return logits, None, None, None, None, None, None, enc

    region_heatmap_logits = model.region_heatmap_logits(q_last)
    semantic_class_logits = model.semantic_class_logits(q_last)
    side_preference_logits = model.side_preference_logits(q_last)
    subzone_logits = model.subzone_logits(q_last)

    pairwise_logits = None
    if enc.shape[1] > 1:
        peer_enc_all = enc[:, :-1, :]
        if token_key_padding_mask is None:
            peer_enc = peer_enc_all.mean(dim=1)
        else:
            peer_valid = (~token_key_padding_mask[:, :-1]).to(peer_enc_all.dtype).unsqueeze(-1)
            denom = peer_valid.sum(dim=1).clamp(min=1.0)
            peer_enc = (peer_enc_all * peer_valid).sum(dim=1) / denom
        pairwise_logits = model.pairwise_relation_logits(q_last, peer_enc)

    region_prior_rows: List[torch.Tensor] = []
    for j, (env, ref) in enumerate(zip(envs, refs)):
        region_idx = _cached_region_indices_cuda(
            env,
            ref,
            region_cfg,
            device=region_heatmap_logits.device,
        )
        prior = action_region_prior_from_predictions_cuda(
            region_heatmap_logits[j],
            action_heatmap_cell_idx_flat=region_idx,
            heatmap_action_prior_weight=float(region_cfg.heatmap_action_prior_weight),
        )
        region_prior_rows.append(prior.reshape(-1))
    region_prior = _pad_1d_tensors(region_prior_rows, pad_value=0.0)
    return logits, region_heatmap_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior, enc


def _region_aux_losses_batch(
    *,
    refs: List[str],
    task_paths: List[str],
    envs: List[PlacementEnv],
    region_cfg: RegionPriorConfig,
    region_heatmap_logits: Optional[torch.Tensor],
    semantic_class_logits: Optional[torch.Tensor],
    side_preference_logits: Optional[torch.Tensor],
    subzone_logits: Optional[torch.Tensor],
    pairwise_logits: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched auxiliary region losses and split supervision weights. Output tensors are [B].

    The previous implementation executed the heatmap CE, semantic CE, side CE,
    subzone CE, and pairwise BCE one row at a time.  Targets still have to be
    fetched per ref because they come from per-board JSON/runtime caches, but the
    differentiable loss math is now run as row-wise batched tensor ops on the
    device.  This cuts Python/autograd overhead in the main training and replay
    update loops without changing any target semantics.
    """
    B = len(envs)
    if (not bool(region_cfg.enabled)) or region_heatmap_logits is None or semantic_class_logits is None:
        z = torch.zeros((B,), device=device)
        ones = torch.ones((B,), device=device)
        return z, z, z, z, z, z, ones, ones, ones, ones, ones

    # Zero tensors with the correct dtype/device and a graph connection to model
    # outputs.  Rows without target metadata keep zero auxiliary loss.
    dtype = region_heatmap_logits.dtype
    loss_semantic = semantic_class_logits.sum(dim=1) * 0.0
    loss_region_heatmap = region_heatmap_logits.sum(dim=1) * 0.0
    loss_prior_consistency = region_heatmap_logits.sum(dim=1) * 0.0
    loss_side = loss_region_heatmap.clone()
    if side_preference_logits is not None:
        loss_side = side_preference_logits.sum(dim=1) * 0.0
    loss_subzone = loss_region_heatmap.clone()
    if subzone_logits is not None:
        loss_subzone = subzone_logits.sum(dim=1) * 0.0
    loss_pairwise = loss_region_heatmap.clone()
    if pairwise_logits is not None:
        loss_pairwise = pairwise_logits.sum(dim=1) * 0.0

    expert_weights = torch.ones((B,), device=device, dtype=torch.float32)
    semantic_weights = torch.ones((B,), device=device, dtype=torch.float32)
    teacher_weights = torch.ones((B,), device=device, dtype=torch.float32)
    metric_weights = torch.ones((B,), device=device, dtype=torch.float32)
    prior_supervision_weights = torch.ones((B,), device=device, dtype=torch.float32)

    valid_rows: List[int] = []
    region_targets: List[torch.Tensor] = []
    prior_region_targets: List[torch.Tensor] = []
    semantic_targets: List[torch.Tensor] = []
    side_targets: List[torch.Tensor] = []
    subzone_targets: List[torch.Tensor] = []
    prior_weights: List[float] = []
    pairwise_targets: List[torch.Tensor] = []

    for j, (ref, task_path, env) in enumerate(zip(refs, task_paths, envs)):
        target_tensors = _cached_region_target_tensors(env, ref, task_path, region_cfg, device)
        if target_tensors is None:
            continue

        region_t, prior_region_t, semantic_t, side_t, subzone_t, supervision_weights, prior_weight = target_tensors
        valid_rows.append(int(j))
        region_targets.append(region_t.reshape(1, -1))
        prior_region_targets.append(prior_region_t.reshape(1, -1))
        semantic_targets.append(semantic_t.reshape(-1)[:1])
        side_targets.append(side_t.reshape(-1)[:1])
        subzone_targets.append(subzone_t.reshape(-1)[:1])
        prior_weights.append(float(prior_weight))

        expert_weights[j] = float(supervision_weights.expert)
        semantic_weights[j] = float(supervision_weights.semantic)
        teacher_weights[j] = float(supervision_weights.teacher)
        metric_weights[j] = float(supervision_weights.metric)
        prior_supervision_weights[j] = float(supervision_weights.prior)

        if pairwise_logits is not None:
            pairwise_targets.append(_gated_pairwise_relation_target(env, ref, list(env.placed_order), device))

    if valid_rows:
        idx = torch.tensor(valid_rows, device=device, dtype=torch.long)
        region_target_batch = torch.cat(region_targets, dim=0).to(device=device, dtype=dtype)
        region_target_batch = region_target_batch / region_target_batch.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        region_logp = torch.log_softmax(region_heatmap_logits.index_select(0, idx), dim=-1)
        loss_region_heatmap = loss_region_heatmap.index_copy(
            0,
            idx,
            -(region_target_batch * region_logp).sum(dim=-1),
        )

        prior_target_batch = torch.cat(prior_region_targets, dim=0).to(device=device, dtype=dtype)
        prior_target_batch = prior_target_batch / prior_target_batch.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        prior_logp = region_logp
        prior_weight_t = torch.tensor(prior_weights, device=device, dtype=dtype)
        loss_prior_consistency = loss_prior_consistency.index_copy(
            0,
            idx,
            prior_weight_t * (-(prior_target_batch * prior_logp).sum(dim=-1)),
        )

        semantic_t = torch.cat(semantic_targets, dim=0).to(device=device, dtype=torch.long)
        loss_semantic = loss_semantic.index_copy(
            0,
            idx,
            F.cross_entropy(
                semantic_class_logits.index_select(0, idx),
                semantic_t,
                reduction="none",
            ),
        )

        if side_preference_logits is not None:
            side_t = torch.cat(side_targets, dim=0).to(device=device, dtype=torch.long)
            loss_side = loss_side.index_copy(
                0,
                idx,
                F.cross_entropy(
                    side_preference_logits.index_select(0, idx),
                    side_t,
                    reduction="none",
                ),
            )

        if subzone_logits is not None:
            subzone_t = torch.cat(subzone_targets, dim=0).to(device=device, dtype=torch.long)
            loss_subzone = loss_subzone.index_copy(
                0,
                idx,
                F.cross_entropy(
                    subzone_logits.index_select(0, idx),
                    subzone_t,
                    reduction="none",
                ),
            )

        if pairwise_logits is not None and pairwise_targets:
            pairwise_target_batch = torch.cat(pairwise_targets, dim=0).to(device=device, dtype=pairwise_logits.dtype)
            pairwise_loss_batch = F.binary_cross_entropy_with_logits(
                pairwise_logits.index_select(0, idx),
                pairwise_target_batch,
                reduction="none",
            ).mean(dim=-1)
            loss_pairwise = loss_pairwise.index_copy(0, idx, pairwise_loss_batch)

    return (
        loss_semantic,
        loss_region_heatmap,
        loss_prior_consistency,
        loss_side,
        loss_subzone,
        loss_pairwise,
        expert_weights,
        semantic_weights,
        teacher_weights,
        metric_weights,
        prior_supervision_weights,
    )

@dataclass
class BoardTrainingOutcome:
    """Per-board result from one batched suffix-training visit."""

    loss: Optional[float]
    valid_steps: int
    failed: bool
    failure_reason: Optional[str]
    suffix_complete: bool
    terminated: bool

    @property
    def has_trainable_loss(self) -> bool:
        """True when this board contributed a differentiable loss term.

        This is intentionally weaker than ``curriculum_valid``: failed or
        incomplete suffixes can still have useful supervised/teacher prefix
        losses that were already backpropagated and should therefore allow the
        optimizer step.  Curriculum advancement remains strict below.
        """
        return bool(self.loss is not None and self.valid_steps > 0)

    @property
    def curriculum_valid(self) -> bool:
        return bool(
            self.has_trainable_loss
            and not self.failed
            and self.suffix_complete
            and not self.terminated
        )


def _batch_has_trainable_loss(outcomes: List[BoardTrainingOutcome]) -> bool:
    """Return True when a batch produced any differentiable training signal.

    Failed suffix rollouts are still allowed to update the model when they
    collected valid supervised/teacher prefix losses before failure.  This
    helper deliberately does not use ``curriculum_valid`` because curriculum
    promotion and optimizer stepping have different acceptance criteria.
    """
    return any(bool(outcome.has_trainable_loss) for outcome in outcomes)


def _step_optimizer_if_trainable_batch(
    model: nn.Module,
    opt: Any,
    outcomes: List[BoardTrainingOutcome],
    *,
    max_grad_norm: float = 1.0,
) -> bool:
    """Clip gradients and step the optimizer if this batch has trainable loss.

    Returns True when ``opt.step()`` was executed.  Keeping the gate in one
    small helper gives the P0 invariant a direct unit-test target.
    """
    if not _batch_has_trainable_loss(outcomes):
        return False
    nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
    opt.step()
    return True


def rollout_suffix_loss_batched_boards(
    model: MaskedPolicy,
    envs: List[PlacementEnv],
    expert_actions_list: List[List[Tuple[int, int, int]]],
    task_paths: List[str],
    k_list: List[int],
    alpha_expert: float,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    device: torch.device,
    *,
    max_tokens: int = 128,
    do_backward: bool = True,
    grad_weight: float = 1.0,
    prefix_replayed: bool = False,
    window_starts: Optional[List[int]] = None,
    window_steps: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[BoardTrainingOutcome]]:
    """True multi-board batched GPU suffix loss.

    Per suffix position this builds the real batched tensors requested by the
    training path:
      tokens_batch      [B_active, L, D]
      action_feat_batch [B_active, Amax, F]
      mask_batch        [B_active, Amax]
      objective_batch   [B_active, Amax]
      logits_batch      [B_active, Amax]
      loss_batch        [B_active]

    Different boards may have different action-space sizes; rows are padded to
    Amax and padding is masked out before softmax/cross-entropy/rollout.
    Environment stepping is still sequential after the batched GPU decision.
    JSON parsing, env construction, and expert-prefix replay can be skipped by
    passing envs cloned from _BoardRuntimeCache with prefix_replayed=True.
    """
    B = len(envs)
    if B <= 0:
        return torch.tensor(0.0, device=device), []
    if not (B == len(expert_actions_list) == len(task_paths) == len(k_list)):
        raise ValueError("envs, expert_actions_list, task_paths, and k_list must have the same length")

    T_list = [len(env.sequence) for env in envs]

    if window_starts is None:
        # Backward-compatible behavior: train the final k actions.
        prefix_lens = [max(0, int(T) - int(k)) for T, k in zip(T_list, k_list)]
    else:
        if len(window_starts) != B:
            raise ValueError("window_starts length must match batch size")
        prefix_lens = [
            max(0, min(int(T), int(start)))
            for T, start in zip(T_list, window_starts)
        ]

    if window_steps is None:
        window_ends = list(T_list)
    else:
        if len(window_steps) != B:
            raise ValueError("window_steps length must match batch size")
        window_ends = [
            max(
                int(prefix_lens[i]),
                min(int(T_list[i]), int(prefix_lens[i]) + max(1, int(window_steps[i]))),
            )
            for i in range(B)
        ]

    suffix_steps = [
        max(1, int(window_ends[i]) - int(prefix_lens[i]))
        for i in range(B)
    ]

    if prefix_replayed:
        cur_t = [int(env.t) for env in envs]
        for i, (got, expected) in enumerate(zip(cur_t, prefix_lens)):
            if got != expected:
                raise ValueError(
                    f"Cached prefix mismatch for board={task_paths[i]}: "
                    f"env.t={got}, expected prefix_len={expected}"
                )
    else:
        cur_t = list(prefix_lens)
        for i, env in enumerate(envs):
            expert_actions = expert_actions_list[i]
            for t in range(prefix_lens[i]):
                ref_before = env.current_ref() if not env.done() else None
                _, _, _, info = env.step(expert_actions[t], assume_legal=True, return_observation=False, compute_objective=False)
                if info.get("illegal"):
                    raise ValueError(f"Illegal expert action at t={t} ref={ref_before} info={info}")
            cur_t[i] = prefix_lens[i]

    per_board_sum = [0.0 for _ in range(B)]
    per_board_cnt = [0 for _ in range(B)]
    failure_reasons: List[Optional[str]] = [None for _ in range(B)]
    per_board_loss_terms: List[List[torch.Tensor]] = [[] for _ in range(B)]

    def _mark_board_failure(index: int, reason: str) -> None:
        i = int(index)
        if failure_reasons[i] is None:
            failure_reasons[i] = str(reason)
        envs[i].terminated = True
        cur_t[i] = window_ends[i]

    while True:
        candidate_indices = [i for i in range(B) if (cur_t[i] < window_ends[i]) and (not envs[i].done())]
        if not candidate_indices:
            break

        valid_indices: List[int] = []
        refs: List[str] = []
        active_envs: List[PlacementEnv] = []
        active_paths: List[str] = []
        masks: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        objectives_teacher: List[torch.Tensor] = []
        objectives_metric: List[torch.Tensor] = []
        tokens: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        action_priors: List[torch.Tensor] = []
        action_lengths: List[int] = []
        gt_indices: List[int] = []
        expert_gt_actions: List[Tuple[int, int, int]] = []

        candidate_envs = [envs[i] for i in candidate_indices]
        candidate_refs = [envs[i].current_ref() for i in candidate_indices]
        candidate_maps = _batch_objective_and_mask(candidate_envs, candidate_refs, device)

        for local_row, i in enumerate(candidate_indices):
            env = envs[i]
            ref = candidate_refs[local_row]
            mask_map_t, bias_map_t, objective_teacher_map_t, objective_metric_map_t = candidate_maps[local_row]
            mask_t = mask_map_t.reshape(-1)
            bias_t = bias_map_t.reshape(-1)

            if not bool((mask_t > 0.5).any().item()):
                _mark_board_failure(i, "no_legal_action")
                continue

            objective_teacher_t = objective_teacher_map_t.reshape(-1)
            objective_metric_t = objective_metric_map_t.reshape(-1)
            tokens_t = build_context_tokens_cuda(env, ref, device, max_tokens=max_tokens)
            feat_t = _action_features_for_ref_cuda(env, ref, device)
            action_prior_t = _action_prior_for_ref_cuda(env, ref, mask_t, teacher, device)

            w, h = env.grid_shape()
            R = len(env.rotations)
            A = R * w * h
            gt = expert_actions_list[i][cur_t[i]]
            gt_a = _flatten_action(gt[0], gt[1], gt[2], w, h)
            if gt_a < 0 or gt_a >= A or not bool((mask_t[gt_a] > 0.5).item()):
                legal_idx = torch.where(mask_t > 0.5)[0]
                if legal_idx.numel() <= 0:
                    _mark_board_failure(i, "no_legal_action")
                    continue
                gt_a = int(legal_idx[torch.argmax(bias_t[legal_idx])].item())

            valid_indices.append(i)
            refs.append(ref)
            active_envs.append(env)
            active_paths.append(task_paths[i])
            masks.append(mask_t)
            biases.append(bias_t)
            objectives_teacher.append(objective_teacher_t)
            objectives_metric.append(objective_metric_t)
            tokens.append(tokens_t)
            feats.append(feat_t)
            action_priors.append(action_prior_t)
            action_lengths.append(int(A))
            gt_indices.append(int(gt_a))
            expert_gt_actions.append(gt)

        if not valid_indices:
            continue

        tokens_batch, token_key_padding_mask = _left_pad_context_tokens(tokens, pad_value=0.0)
        action_feat_batch = _pad_2d_tensors(feats, pad_value=0.0)
        mask_batch = _pad_1d_tensors(masks, pad_value=0.0)
        bias_batch = _pad_1d_tensors(biases, pad_value=0.0)
        objective_teacher_batch = _pad_1d_tensors(objectives_teacher, pad_value=0.0)
        objective_metric_batch = _pad_1d_tensors(objectives_metric, pad_value=0.0)
        action_prior_batch = _pad_1d_tensors(action_priors, pad_value=0.0)

        policy_logits_batch, region_heatmap_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior_batch, _enc = _policy_outputs_with_region_batch(
            model,
            active_envs,
            refs,
            tokens_batch,
            action_feat_batch,
            region_cfg,
            token_key_padding_mask=token_key_padding_mask,
        )
        # Bias remains part of the supervised logits. Rollout selection below
        # uses the same shared objective-aware scorer as replay and inference.
        logits_batch = (
            policy_logits_batch + bias_batch
        ).masked_fill(mask_batch < 0.5, -1e9)

        (
            loss_semantic, loss_region_heatmap, loss_prior_consistency, loss_side, loss_subzone, loss_pairwise,
            expert_weight, semantic_weight, teacher_weight, metric_weight, prior_supervision_weight,
        ) = _region_aux_losses_batch(
            refs=refs,
            task_paths=active_paths,
            envs=active_envs,
            region_cfg=region_cfg,
            region_heatmap_logits=region_heatmap_logits,
            semantic_class_logits=semantic_class_logits,
            side_preference_logits=side_preference_logits,
            subzone_logits=subzone_logits,
            pairwise_logits=pairwise_logits,
            device=device,
        )
        expert_weight = torch.clamp(expert_weight, min=0.0, max=1.0)
        semantic_weight = torch.clamp(semantic_weight, min=0.0, max=1.0)
        teacher_weight = torch.clamp(teacher_weight, min=0.0, max=1.0)
        metric_weight = torch.clamp(metric_weight, min=0.0, max=1.0)
        prior_supervision_weight = torch.clamp(prior_supervision_weight, min=0.0, max=1.0)
        region_prior_for_teacher = None if region_prior_batch is None else region_prior_batch * teacher_weight.unsqueeze(1)

        q_batch, cand_batch = _build_teacher_distribution_batch_cuda(
            mask_batch,
            objective_teacher_batch,
            teacher,
            device=device,
            region_batch=region_prior_for_teacher,
            action_prior_batch=action_prior_batch,
        )
        logp_batch = torch.log_softmax(logits_batch, dim=-1)
        loss_teacher = -(q_batch * logp_batch).sum(dim=-1)
        probs_batch = torch.softmax(logits_batch, dim=-1)
        # Teacher CE uses residual_total; metric loss uses full total so all objective_* weights train the policy.
        loss_metric = (probs_batch * objective_metric_batch).sum(dim=-1)

        gt_t = torch.tensor(gt_indices, device=device, dtype=torch.long)
        loss_expert = F.cross_entropy(logits_batch, gt_t, reduction="none")

        geometry_gt_indices = [
            _flatten_action(int(a[0]), int(a[1]), int(a[2]), *envs[i].grid_shape())
            for i, a in zip(valid_indices, expert_gt_actions)
        ]
        loss_geo = _geometry_distill_loss_batch(
            envs=active_envs,
            refs=refs,
            logits_batch=logits_batch,
            action_feat_batch=action_feat_batch,
            geometry_gt_indices=geometry_gt_indices,
            geom_cfg=geom_cfg,
        )

        aexp = float(alpha_expert)
        # Split trust streams: semantic review gates only semantic aux labels. It no
        # longer implicitly disables expert placement, teacher CE, or metric loss.
        loss_batch = (
            expert_weight * aexp * loss_expert
            + teacher_weight * (1.0 - aexp) * loss_teacher
            + metric_weight * float(teacher.metric_weight) * loss_metric
            + expert_weight * aexp * loss_geo
            + expert_weight * float(region_cfg.aux_heatmap_weight) * loss_region_heatmap
            + prior_supervision_weight * float(region_cfg.aux_prior_consistency_weight) * loss_prior_consistency
            + semantic_weight * float(region_cfg.aux_semantic_weight) * loss_semantic
            + semantic_weight * float(region_cfg.aux_side_weight) * loss_side
            + semantic_weight * float(region_cfg.aux_subzone_weight) * loss_subzone
            + semantic_weight * float(region_cfg.aux_pairwise_weight) * loss_pairwise
        )

        for row, i in enumerate(valid_indices):
            per_board_sum[i] += float(loss_batch[row].detach().item())
            per_board_cnt[i] += 1

        weights = torch.tensor(
            [float(grad_weight) / (float(B) * float(max(1, suffix_steps[i]))) for i in valid_indices],
            device=device,
            dtype=loss_batch.dtype,
        )
        weighted_loss_batch = loss_batch * weights
        for row, i in enumerate(valid_indices):
            per_board_loss_terms[i].append(weighted_loss_batch[row])

        rollout_score_batch, _rollout_candidates = score_actions_batched(
            policy_logits_batch,
            bias_batch,
            mask_batch,
            objective_teacher_batch,
            objective_metric_batch,
            region_prior_batch,
            action_prior_batch,
            teacher=teacher,
            objective_alpha=DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
            region_alpha=DEFAULT_ROLLOUT_REGION_ALPHA,
        )
        actions = torch.argmax(
            rollout_score_batch,
            dim=-1,
        ).detach().cpu().tolist()

        for row, i in enumerate(valid_indices):
            a = int(actions[row])
            A_i = int(action_lengths[row])
            if a < 0 or a >= A_i:
                # Should not happen because padding is masked, but keep a safe fallback.
                legal_idx = torch.where(masks[row] > 0.5)[0]
                if legal_idx.numel() <= 0:
                    _mark_board_failure(i, "no_legal_action_after_policy")
                    continue
                a = int(legal_idx[torch.argmax(biases[row][legal_idx])].item())
            w, h = envs[i].grid_shape()
            R = len(envs[i].rotations)
            selected_action = _unflatten_action(a, w, h)
            expert_action = expert_actions_list[i][cur_t[i]]
            selected_action = _choose_dead_end_safe_action(
                envs[i],
                selected_action,
                expert_action,
                rollout_score_batch[row, :A_i],
                masks[row],
                w,
                h,
                device=device,
            )
            _, _, _, info = envs[i].step(selected_action, assume_legal=True, return_observation=False, compute_objective=False)
            cur_t[i] += 1
            if info.get("illegal"):
                _mark_board_failure(
                    i,
                    f"illegal_step:{info.get('reason') or 'unknown'}",
                )

    # Keep every valid supervised/teacher loss term collected before a rollout
    # failure.  A late illegal/no-legal suffix should not erase the useful prefix
    # gradient for the same board.  Curriculum advancement below still requires
    # suffix_complete=True, so failed boards do not get promoted just because a
    # prefix produced loss.
    valid_loss_terms: List[torch.Tensor] = []
    for i in range(B):
        if per_board_loss_terms[i]:
            valid_loss_terms.extend(per_board_loss_terms[i])

    if valid_loss_terms:
        total = torch.stack(valid_loss_terms).sum()
        if do_backward:
            total.backward()
    else:
        total = torch.tensor(0.0, device=device)

    outcomes: List[BoardTrainingOutcome] = []
    for i in range(B):
        valid_steps = int(per_board_cnt[i])
        loss_value = (
            float(per_board_sum[i] / float(valid_steps))
            if valid_steps > 0
            else None
        )
        suffix_complete = bool(
            failure_reasons[i] is None
            and not envs[i].terminated
            and int(cur_t[i]) >= int(window_ends[i])
        )
        failure_reason = failure_reasons[i]
        if failure_reason is None and valid_steps <= 0:
            failure_reason = "no_valid_loss"
        elif failure_reason is None and not suffix_complete:
            failure_reason = "incomplete_suffix"

        outcomes.append(
            BoardTrainingOutcome(
                loss=loss_value,
                valid_steps=valid_steps,
                failed=failure_reason is not None,
                failure_reason=failure_reason,
                suffix_complete=suffix_complete,
                terminated=bool(envs[i].terminated),
            )
        )
    return total, outcomes


# Backward-compatible name. This is now a real batched implementation; the old
# function only looped over boards and accumulated losses.
def rollout_suffix_loss_batch(
    model: MaskedPolicy,
    envs: List[PlacementEnv],
    expert_actions_list: List[List[Tuple[int, int, int]]],
    task_paths: List[str],
    k_list: List[int],
    alpha_expert: float,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    device: torch.device,
    *,
    max_tokens: int = 128,
    do_backward: bool = True,
    grad_weight: float = 1.0,
    prefix_replayed: bool = False,
) -> Tuple[torch.Tensor, List[BoardTrainingOutcome]]:
    return rollout_suffix_loss_batched_boards(
        model=model,
        envs=envs,
        expert_actions_list=expert_actions_list,
        task_paths=task_paths,
        k_list=k_list,
        alpha_expert=alpha_expert,
        teacher=teacher,
        region_cfg=region_cfg,
        geom_cfg=geom_cfg,
        device=device,
        max_tokens=max_tokens,
        do_backward=do_backward,
        grad_weight=grad_weight,
        prefix_replayed=prefix_replayed,
    )


def _batch_objective_and_mask(
    envs: List[PlacementEnv],
    refs: List[str],
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute legality masks, bias maps, and objective maps in a stable order.

    Boards are grouped by grid shape to improve locality and leave room for
    future true tensor batching.  The expensive work is still delegated to the
    existing CUDA kernels per board, but this wrapper keeps all target-map
    creation under no_grad and preserves the input order exactly.
    """
    results: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = [None] * len(envs)
    grid_groups: Dict[Tuple[int, int], List[int]] = {}
    for i, env in enumerate(envs):
        grid_groups.setdefault(tuple(env.grid_shape()), []).append(i)

    with torch.no_grad():
        for _grid_key, indices in grid_groups.items():
            for i in indices:
                env = envs[i]
                ref = refs[i]
                mask_map, bias_map = action_mask_and_bias_cuda(env, ref, device)
                obj_maps = objective_delta_mask_cuda(env, ref, device)
                results[i] = (mask_map, bias_map, obj_maps.get("residual_total", obj_maps["total"]), obj_maps["total"])

    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for r in results:
        if r is None:
            raise RuntimeError("_batch_objective_and_mask internal ordering error")
        out.append(r)
    return out



# -------------------------
# Weighted Replay Fine-tune
# -------------------------
REPLAY_SCORE_KIND = "relative_expert_improvement_v1"
_REPLAY_EXP_CLIP = 20.0
_REPLAY_BATCH_EXP_CLIP = 60.0


def _replay_relative_score(final_obj: float, reference_obj: float) -> float:
    """Return a board-scale-normalized score where higher is better.

    The score is the fractional objective improvement over the expert/reference
    layout for the same board. Expert-equivalent performance is 0, a 10 percent
    improvement is +0.1, and a 10 percent degradation is -0.1.
    """
    final_value = float(final_obj)
    reference_value = float(reference_obj)
    if not math.isfinite(final_value) or not math.isfinite(reference_value):
        raise ValueError(
            f"Replay objectives must be finite; final_obj={final_obj!r}, "
            f"reference_obj={reference_obj!r}"
        )
    denominator = max(abs(reference_value), 1.0)
    return float((reference_value - final_value) / denominator)


def _replay_priority_from_score(score: float, replay_temp: float) -> float:
    """Map a relative replay score to a finite positive sampling priority."""
    temperature = max(1e-6, float(replay_temp))
    scaled = float(np.clip(float(score) / temperature, -_REPLAY_EXP_CLIP, _REPLAY_EXP_CLIP))
    priority = float(math.exp(scaled))
    if not math.isfinite(priority) or priority <= 0.0:
        return 1e-8
    return max(1e-8, priority)


def _stable_replay_batch_weights(
    scores: np.ndarray,
    replay_temp: float,
) -> np.ndarray:
    """Return finite, strictly positive mean-one weights without underflow.

    Subtracting the maximum keeps the best sample at exp(0)=1. Clipping the
    lower tail prevents every other sample from becoming exact zero.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Replay scores must be finite; got {values.tolist()}")

    temperature = max(1e-6, float(replay_temp))
    scaled = values / temperature
    scaled = scaled - float(np.max(scaled))
    scaled = np.clip(scaled, -_REPLAY_BATCH_EXP_CLIP, 0.0)
    weights = np.exp(scaled)
    mean_weight = float(weights.mean())
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        return np.ones_like(values, dtype=np.float64)
    weights = weights / mean_weight
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        return np.ones_like(values, dtype=np.float64)
    return weights


def _compute_replay_advantages_and_update_baselines(
    episodes: List[Dict[str, Any]],
    baselines_by_path: Dict[str, float],
    *,
    baseline_beta: float,
    advantage_clip: float,
) -> List[float]:
    """Compute per-board REINFORCE advantages and update EMA baselines.

    All episodes for the same board in one iteration use the same pre-update
    baseline, which avoids order-dependent advantages.
    """
    beta = float(min(0.999999, max(0.0, baseline_beta)))
    clip_value = float(max(0.0, advantage_clip))

    _require_normalized_replay_episodes(
        episodes,
        context="replay policy-gradient advantage",
    )
    scores_by_path: Dict[str, List[float]] = {}
    advantages: List[float] = []
    for episode in episodes:
        path = str(episode["task_path"])
        score = float(episode["score"])
        if not math.isfinite(score):
            raise ValueError(
                f"Replay policy-gradient score must be finite; path={path!r}, score={score!r}"
            )
        baseline = float(baselines_by_path.get(path, 0.0))
        advantage = score - baseline
        if clip_value > 0.0:
            advantage = float(np.clip(advantage, -clip_value, clip_value))
        advantages.append(float(advantage))
        scores_by_path.setdefault(path, []).append(score)

    for path, scores in scores_by_path.items():
        batch_mean = float(np.mean(np.asarray(scores, dtype=np.float64)))
        old = float(baselines_by_path.get(path, 0.0))
        baselines_by_path[path] = beta * old + (1.0 - beta) * batch_mean
    return advantages


def _summarize_replay_rollout_episodes(
    episodes: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Compute replay rollout metrics over every sampled rollout.

    This intentionally includes zero-step failures such as first-step
    ``no_legal_action``. Those episodes cannot contribute to policy-gradient
    loss because there is no sampled action log-probability, but excluding them
    from rollout metrics makes complete/no-legal rates look overly optimistic.
    """
    if not episodes:
        return {
            "complete_rate": None,
            "illegal_rate": None,
            "no_legal_rate": None,
            "raw_objective_mean": None,
        }

    denom = float(max(1, len(episodes)))
    return {
        "complete_rate": float(
            sum(1 for ep in episodes if bool(ep.get("complete", False)))
            / denom
        ),
        "illegal_rate": float(
            sum(
                1 for ep in episodes
                if str(ep.get("failure_reason") or "").startswith("illegal_action")
            ) / denom
        ),
        "no_legal_rate": float(
            sum(
                1 for ep in episodes
                if "no_legal" in str(ep.get("failure_reason") or "")
            ) / denom
        ),
        "raw_objective_mean": float(
            np.mean([
                float(ep.get("partial_obj", ep.get("final_obj", 0.0)))
                for ep in episodes
            ])
        ),
    }


def _validate_normalized_replay_episode(
    episode: Dict[str, Any],
    *,
    context: str,
) -> None:
    """Fail fast if a replay item is not in the canonical score space."""
    if episode.get("score_kind") != REPLAY_SCORE_KIND:
        raise ValueError(
            f"{context}: replay episode must use score_kind={REPLAY_SCORE_KIND!r}; "
            f"got {episode.get('score_kind')!r} for task_path={episode.get('task_path')!r}"
        )
    for key in ("score", "final_obj", "reference_obj", "raw_score"):
        value = float(episode.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(
                f"{context}: replay episode field {key!r} must be finite; "
                f"task_path={episode.get('task_path')!r}, value={episode.get(key)!r}"
            )


def _require_normalized_replay_episodes(
    episodes: List[Dict[str, Any]],
    *,
    context: str,
) -> List[Dict[str, Any]]:
    for episode in episodes:
        _validate_normalized_replay_episode(
            episode,
            context=context,
        )
    return episodes


def _normalize_replay_episode_for_path(
    episode: Dict[str, Any],
    reference_obj_by_path: Dict[str, float],
    *,
    context: str,
) -> Dict[str, Any]:
    task_path = str(episode.get("task_path", ""))
    if task_path not in reference_obj_by_path:
        raise ValueError(
            f"{context}: missing replay reference objective for task_path={task_path!r}"
        )
    normalized = _normalize_replay_episode_score(
        episode,
        reference_obj_by_path[task_path],
    )
    _validate_normalized_replay_episode(
        normalized,
        context=context,
    )
    return normalized


def _normalize_replay_episode_score(
    episode: Dict[str, Any],
    reference_obj: float,
) -> Dict[str, Any]:
    """Attach the canonical normalized replay score to an episode.

    Legacy replay items stored ``score=-final_obj`` and some expert bootstrap
    items did not store ``final_obj``. Those items are migrated here.
    """
    out = dict(episode)
    final_obj_raw = out.get("final_obj")
    if final_obj_raw is None:
        score_raw = float(out.get("score", 0.0))
        if out.get("score_kind") == REPLAY_SCORE_KIND:
            denominator = max(abs(float(reference_obj)), 1.0)
            final_obj_raw = float(reference_obj) - score_raw * denominator
        else:
            final_obj_raw = -score_raw

    final_obj = float(final_obj_raw)
    relative_score = _replay_relative_score(final_obj, float(reference_obj))
    out["final_obj"] = final_obj
    out["raw_score"] = -final_obj
    out["reference_obj"] = float(reference_obj)
    out["score"] = relative_score
    out["score_kind"] = REPLAY_SCORE_KIND
    return out


class WeightedReplayBuffer:
    def __init__(self, capacity: int = 2000, alpha: float = 0.7):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.items: List[Dict[str, Any]] = []
        self.priorities: List[float] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: Dict[str, Any], priority: float) -> None:
        _validate_normalized_replay_episode(
            item,
            context="WeightedReplayBuffer.add",
        )
        p = float(max(1e-8, priority))
        if len(self.items) >= self.capacity:
            self.items.pop(0)
            self.priorities.pop(0)
        self.items.append(item)
        self.priorities.append(p)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.items:
            return []
        ps = np.asarray(self.priorities, dtype=np.float64)
        ps = np.where(np.isfinite(ps) & (ps > 0.0), ps, 1e-8)
        ps = ps ** self.alpha
        total = float(ps.sum())
        if not math.isfinite(total) or total <= 0.0:
            ps = np.full((len(self.items),), 1.0 / float(len(self.items)), dtype=np.float64)
        else:
            ps = ps / total
        idx = np.random.choice(
            len(self.items),
            size=min(batch_size, len(self.items)),
            replace=False,
            p=ps,
        )
        return [self.items[int(i)] for i in idx]


@dataclass
class BoardCurriculumState:
    index: int
    path: str
    kmax: int
    k_stage: int = 1
    k_stage_visits: int = 0
    k_stage_loss_ema: Optional[float] = None
    k_stage_peak_ema: Optional[float] = None
    k_stage_target_ema: Optional[float] = None
    k_stage_target_locked: bool = False
    k_stage_probe_left: int = 0
    k_stage_ready: bool = False
    k_stage_deferred: bool = False
    k_stage_defer_reason: Optional[str] = None
    graduated: bool = False
    full_visits: int = 0
    full_loss_ema: Optional[float] = None
    full_peak_ema: Optional[float] = None
    full_target_ema: Optional[float] = None
    full_probe_left: int = 0
    recent_full_ema: List[float] = field(default_factory=list)
    graduate_step: Optional[int] = None
    graduate_reason: Optional[str] = None


@dataclass
class _CachedBoardRuntime:
    """Immutable-ish per-board runtime objects plus expert-prefix snapshots.

    In the tail-masked objective, a sample always starts from the deterministic
    expert prefix t = T-k and only the suffix is model-rolled out.  The prefix
    therefore should be built once per board and cloned, not replayed every
    training visit.
    """
    index: int
    path: str
    task: Any
    base_env: PlacementEnv
    expert_actions: List[Tuple[int, int, int]]
    prefix_states: List[Dict[str, Any]] = field(default_factory=list)
    action_features_by_device: Dict[str, torch.Tensor] = field(default_factory=dict)
    region_indices_by_key: Dict[Tuple[str, str, int, int, float, float], torch.Tensor] = field(default_factory=dict)
    region_targets_by_cfg: Dict[Tuple[int, int, float, float, float], Dict[str, Any]] = field(default_factory=dict)
    region_target_tensors_by_key: Dict[Tuple[Any, ...], Optional[Tuple[Any, ...]]] = field(default_factory=dict)


def _snapshot_env_dynamic_state(env: PlacementEnv) -> Dict[str, Any]:
    """Copy only mutable episode state; static board/task fields stay shared."""
    return {
        't': int(env.t),
        'terminated': bool(env.terminated),
        'placed': dict(env.placed),
        'placed_order': list(env.placed_order),
        'occupied': list(env.occupied),
        'prev_obj': float(env.prev_obj),
    }


def _clone_env_from_dynamic_state(base_env: PlacementEnv, state: Dict[str, Any]) -> PlacementEnv:
    """Cheap episode clone: share static PlacementEnv fields, copy mutable state."""
    env = copy.copy(base_env)
    env.t = int(state['t'])
    env.terminated = bool(state['terminated'])
    env.placed = dict(state['placed'])
    env.placed_order = list(state['placed_order'])
    env.occupied = list(state['occupied'])
    env.prev_obj = float(state['prev_obj'])
    return env

def _action_keeps_next_mask_nonempty(
    env: PlacementEnv,
    action: Tuple[int, int, int],
    *,
    device: Optional[torch.device] = None,
) -> bool:
    """Return True if taking *action* does not immediately dead-end the next ref.

    This is a cheap one-step rollout guard for tight boards.  The selected action
    is already legal for the current component, but on dense layouts it can still
    consume the only legal placement window for the next large component or the
    last element of a resistor/LED bank.  When a CUDA device is provided, the
    expensive next-ref mask is computed by the CUDA mask path instead of the
    legacy NumPy/CPU environment method.
    """
    state = _snapshot_env_dynamic_state(env)
    probe = _clone_env_from_dynamic_state(env, state)
    _, _, _done, info = probe.step(
        action,
        assume_legal=True,
        return_observation=False,
        compute_objective=False,
    )
    if info.get('illegal'):
        return False
    if probe.done():
        return True
    next_ref = probe.current_ref()
    if device is not None:
        mask_t, _bias_t = action_mask_and_bias_cuda(probe, str(next_ref), device)
        return bool((mask_t > 0.5).any().item())
    mask, _bias = probe.action_mask_and_bias(str(next_ref))
    return bool((mask > 0.5).any())


def _choose_dead_end_safe_action(
    env: PlacementEnv,
    selected_action: Tuple[int, int, int],
    expert_action: Tuple[int, int, int],
    rollout_scores: torch.Tensor,
    legal_mask: torch.Tensor,
    w: int,
    h: int,
    *,
    device: Optional[torch.device] = None,
    max_probe_actions: int = 128,
) -> Tuple[int, int, int]:
    """Prefer the rollout action, but fall back before it creates an empty next mask."""
    if _action_keeps_next_mask_nonempty(env, selected_action, device=device):
        return selected_action
    if _action_keeps_next_mask_nonempty(env, expert_action, device=device):
        return expert_action

    legal_idx = torch.where(legal_mask > 0.5)[0]
    if legal_idx.numel() <= 0:
        return selected_action
    scores = rollout_scores[legal_idx]
    order = torch.argsort(scores, descending=True)
    limit = min(int(max_probe_actions), int(order.numel()))
    for pos in order[:limit].detach().cpu().tolist():
        a2 = int(legal_idx[int(pos)].item())
        candidate = _unflatten_action(a2, w, h)
        if _action_keeps_next_mask_nonempty(env, candidate, device=device):
            return candidate
    return selected_action


class _BoardRuntimeCache:
    """Lazy cache for JSON parsing, PlacementEnv construction, and expert prefixes.

    For every board we precompute snapshots after expert prefix lengths 0..T in
    one forward pass.  A training visit at k then clones prefix_len=T-k instead
    of doing task_from_json()+PlacementEnv()+T-k CPU env.step calls again.
    """

    def __init__(
        self,
        train_tasks: List[Dict[str, Any]],
        env_kwargs: Dict[str, Any],
        *,
        sequence_policy: str = 'rebuild',
    ):
        self.train_tasks = train_tasks
        self.env_kwargs = dict(env_kwargs)
        self.sequence_policy = normalize_sequence_policy(sequence_policy)
        self.path_to_index = {str(it['path']): int(i) for i, it in enumerate(train_tasks)}
        self._items: Dict[int, _CachedBoardRuntime] = {}
        self.json_loads = 0
        self.env_builds = 0
        self.prefix_replays = 0
        self.clones = 0

    def _build_item(self, index: int) -> _CachedBoardRuntime:
        idx = int(index)
        task_item = self.train_tasks[idx]
        path = str(task_item['path'])
        task = task_from_json(path, sequence_policy=self.sequence_policy, load_expert=False)
        self.json_loads += 1
        base_env = PlacementEnv(task, **self.env_kwargs)
        self.env_builds += 1
        expert_actions = list(task_item['expert_actions'])

        env = _clone_env_from_dynamic_state(base_env, _snapshot_env_dynamic_state(base_env))
        prefix_states: List[Dict[str, Any]] = [_snapshot_env_dynamic_state(env)]
        T = len(expert_actions)
        for t, action in enumerate(expert_actions):
            ref_before = env.current_ref() if not env.done() else None
            _, _, _, info = env.step(action, assume_legal=True, return_observation=False, compute_objective=False)
            self.prefix_replays += 1
            if info.get('illegal'):
                raise ValueError(
                    f"Illegal expert prefix action while building cache: "
                    f"file={path} t={t} ref={ref_before} action={action} info={info}"
                )
            prefix_states.append(_snapshot_env_dynamic_state(env))
            if env.done() and t + 1 < T:
                break

        if len(prefix_states) < T + 1:
            last = prefix_states[-1]
            while len(prefix_states) < T + 1:
                prefix_states.append(last)

        item = _CachedBoardRuntime(
            index=idx,
            path=path,
            task=task,
            base_env=base_env,
            expert_actions=expert_actions,
            prefix_states=prefix_states,
        )
        # Lightweight env clones inherit this reference via copy.copy(), allowing
        # GPU-side static tensors (action features, region indices/targets) to be
        # reused across visits instead of rebuilt in every step.
        setattr(base_env, "_runtime_cache_item", item)
        self._items[idx] = item
        return item

    def get_item(self, index: int) -> _CachedBoardRuntime:
        idx = int(index)
        return self._items.get(idx) or self._build_item(idx)

    def get_item_by_path(self, path: str) -> Optional[_CachedBoardRuntime]:
        idx = self.path_to_index.get(str(path))
        if idx is None:
            return None
        return self.get_item(idx)

    def make_fresh_env(self, index: int) -> PlacementEnv:
        item = self.get_item(index)
        env = _clone_env_from_dynamic_state(item.base_env, item.prefix_states[0])
        self.clones += 1
        return env

    def make_fresh_env_by_path(self, path: str) -> Optional[PlacementEnv]:
        item = self.get_item_by_path(path)
        if item is None:
            return None
        env = _clone_env_from_dynamic_state(item.base_env, item.prefix_states[0])
        self.clones += 1
        return env

    def make_env_at_suffix(self, index: int, k_eff: int) -> PlacementEnv:
        item = self.get_item(index)
        T = len(item.expert_actions)
        prefix_len = max(0, min(T, T - int(k_eff)))
        env = _clone_env_from_dynamic_state(item.base_env, item.prefix_states[prefix_len])
        self.clones += 1
        return env

    def make_env_at_prefix(self, index: int, prefix_len: int) -> PlacementEnv:
        """Clone an env after replaying an arbitrary expert prefix length.

        This is used by front-k and random-window curriculum modes.  The name
        means "make an env at prefix length p", not necessarily "prefix-only".
        """
        item = self.get_item(index)
        T = len(item.expert_actions)
        p = max(0, min(T, int(prefix_len)))
        env = _clone_env_from_dynamic_state(item.base_env, item.prefix_states[p])
        self.clones += 1
        return env

    def summary(self) -> str:
        return (
            f"cached_boards={len(self._items)} json_loads={self.json_loads} "
            f"env_builds={self.env_builds} prefix_steps_replayed={self.prefix_replays} "
            f"env_clones={self.clones}"
        )


def _device_cache_key(device: torch.device) -> str:
    """Stable key for tensors cached per device."""
    dev = torch.device(device)
    return f"{dev.type}:{-1 if dev.index is None else int(dev.index)}"


def _runtime_item_from_env(env: PlacementEnv) -> Optional[Any]:
    return getattr(env, "_runtime_cache_item", None)


def _cached_static_action_features_cuda(env: PlacementEnv, device: torch.device) -> torch.Tensor:
    """Cache board-static [x/y/rot] action grid features across env clones.

    Ref/state-conditioned features must be rebuilt after every env.step because
    they depend on current ref, placed anchors, connected centroid, local density,
    and same-module state.  The static grid itself is identical for all clones of
    the same board, so store it on the shared runtime item when available.
    """
    key = (
        _device_cache_key(device),
        tuple(env.grid_shape()),
        tuple(env.rotations),
        tuple(float(v) for v in env.task.bbox_mm),
    )
    runtime_item = _runtime_item_from_env(env)
    if runtime_item is not None:
        cache = getattr(runtime_item, "action_features_by_device", None)
        if isinstance(cache, dict) and key in cache:
            return cache[key]
        feat = action_features_cuda(env, device, ref=None)
        if isinstance(cache, dict):
            cache[key] = feat
        return feat

    cache = getattr(env, "_action_static_feat_cuda_cache", None)
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    feat = action_features_cuda(env, device, ref=None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_action_static_feat_cuda_cache", cache)
    cache[key] = feat
    return feat


def _action_features_for_ref_cuda(env: PlacementEnv, ref: str, device: torch.device) -> torch.Tensor:
    static = _cached_static_action_features_cuda(env, device)
    return action_features_cuda(env, device, ref=ref, static_features=static)


def _action_prior_for_ref_cuda(env: PlacementEnv, ref: str, mask_t: torch.Tensor, teacher: TeacherConfig, device: torch.device) -> torch.Tensor:
    return action_prior_total_cuda(
        env,
        ref,
        device,
        legal_mask_flat=mask_t,
        lambda_conn=float(teacher.lambda_conn_prior),
        lambda_anchor=float(teacher.lambda_anchor_prior),
        lambda_module=float(teacher.lambda_module_prior),
        lambda_spacing=float(teacher.lambda_spacing_prior),
        lambda_edge=float(teacher.lambda_edge_prior),
        lambda_prior_region_heatmap=float(getattr(teacher, 'lambda_prior_region_heatmap', 0.10)),
    )


def _fresh_env_for_path(
    task_path: str,
    env_kwargs: Optional[Dict[str, Any]],
    runtime_cache: Optional["_BoardRuntimeCache"] = None,
) -> PlacementEnv:
    if runtime_cache is not None:
        env = runtime_cache.make_fresh_env_by_path(task_path)
        if env is not None:
            return env
    sequence_policy = (
        runtime_cache.sequence_policy
        if runtime_cache is not None
        else 'rebuild'
    )
    task = task_from_json(task_path, sequence_policy=sequence_policy, load_expert=False)
    return PlacementEnv(task, **(env_kwargs or {}))


def _final_objective(env: PlacementEnv) -> float:
    """Compute final CPU objective once, instead of at every intermediate step."""
    try:
        obj = float(env._objective())
    except Exception:
        obj = float(getattr(env, "prev_obj", 0.0))
    env.prev_obj = obj
    return obj


def _finite_partial_objective(env: PlacementEnv) -> float:
    """Return the current partial-layout objective without using sentinel values."""
    try:
        obj = float(_final_objective(env))
    except Exception:
        obj = float(getattr(env, "prev_obj", 0.0))
    if not math.isfinite(obj):
        obj = 0.0
    env.prev_obj = obj
    return obj


def _failed_rollout_objective_with_penalties(
    *,
    partial_obj: float,
    placed_count: int,
    expected_count: int,
    failure_reason: Optional[str],
    terminated: bool,
) -> Tuple[float, Dict[str, float]]:
    """Shape failed rollout objectives from partial progress plus finite penalties.

    Incomplete on-policy episodes are useful negative-advantage samples, but a
    single 1e30 sentinel swamps per-board baselines and makes every failure look
    identical.  This keeps the score finite and differentiates nearly-complete
    failures from early illegal/no-legal terminations.
    """
    partial_value = float(partial_obj)
    if not math.isfinite(partial_value):
        partial_value = 0.0

    placed = max(0, int(placed_count))
    expected = max(0, int(expected_count))
    missing = max(0, expected - placed)

    component_scale = max(
        float(FAILED_ROLLOUT_MIN_OBJECTIVE_SCALE),
        abs(partial_value) / float(max(1, placed)),
    )
    board_scale = component_scale * float(max(1, expected))
    reason = str(failure_reason or "").lower()

    missing_penalty = (
        float(missing)
        * component_scale
        * float(FAILED_ROLLOUT_MISSING_COMPONENT_PENALTY_SCALE)
    )
    illegal_penalty = 0.0
    if reason.startswith("illegal_action") or reason.startswith("invalid_sampled_action"):
        illegal_penalty = (
            board_scale
            * float(FAILED_ROLLOUT_ILLEGAL_PENALTY_SCALE)
        )
    terminal_penalty = 0.0
    if bool(terminated):
        terminal_penalty = (
            board_scale
            * float(FAILED_ROLLOUT_TERMINAL_PENALTY_SCALE)
        )

    total_penalty = float(missing_penalty + illegal_penalty + terminal_penalty)
    final_obj = float(partial_value + total_penalty)
    if not math.isfinite(final_obj):
        final_obj = float(INCOMPLETE_OBJECTIVE_PENALTY)

    return final_obj, {
        "missing_penalty": float(missing_penalty),
        "illegal_penalty": float(illegal_penalty),
        "terminal_penalty": float(terminal_penalty),
        "total_failure_penalty": float(total_penalty),
        "missing_count": float(missing),
        "objective_scale_per_component": float(component_scale),
    }



@torch.no_grad()
def rollout_episodes_batched(
    model: MaskedPolicy,
    task_items: List[Dict[str, Any]],
    device: torch.device,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    env_kwargs: Optional[Dict[str, Any]] = None,
    runtime_cache: Optional["_BoardRuntimeCache"] = None,
    *,
    max_tokens: int = 128,
    sample_actions: bool = False,
    sampling_temperature: float = 1.0,
) -> List[Dict[str, Any]]:
    """Roll out multiple boards together with one model forward per depth.

    Env mutation remains Python-side, but mask/objective/token construction and
    policy evaluation are grouped across active boards.  Final CPU objective is
    computed once per episode, not once per step.
    """
    if not task_items:
        return []

    paths = [str(it["path"] if isinstance(it, dict) else it) for it in task_items]
    envs = [_fresh_env_for_path(path, env_kwargs, runtime_cache) for path in paths]
    w_h_R = [(env.grid_shape()[0], env.grid_shape()[1], len(env.rotations)) for env in envs]
    actions_flat: List[List[int]] = [[] for _ in envs]
    terminated = [False for _ in envs]
    failure_reasons: List[Optional[str]] = [None for _ in envs]

    while True:
        active = [i for i, env in enumerate(envs) if (not terminated[i]) and (not env.done())]
        if not active:
            break

        active_envs = [envs[i] for i in active]
        refs = [env.current_ref() for env in active_envs]
        maps = _batch_objective_and_mask(active_envs, refs, device)

        valid_rows: List[int] = []
        valid_envs: List[PlacementEnv] = []
        valid_refs: List[str] = []
        masks: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        objectives_teacher: List[torch.Tensor] = []
        objectives_metric: List[torch.Tensor] = []
        tokens: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        action_priors: List[torch.Tensor] = []
        action_lengths: List[int] = []

        for local, i in enumerate(active):
            env = envs[i]
            mask_map_t, bias_map_t, objective_teacher_map_t, objective_metric_map_t = maps[local]
            mask_t = mask_map_t.reshape(-1)
            bias_t = bias_map_t.reshape(-1)
            if not bool((mask_t > 0.5).any().item()):
                env.terminated = True
                terminated[i] = True
                if failure_reasons[i] is None:
                    failure_reasons[i] = "no_legal_action"
                continue
            valid_rows.append(i)
            valid_envs.append(env)
            valid_refs.append(refs[local])
            masks.append(mask_t)
            biases.append(bias_t)
            objectives_teacher.append(objective_teacher_map_t.reshape(-1))
            objectives_metric.append(objective_metric_map_t.reshape(-1))
            tokens.append(build_context_tokens_cuda(env, refs[local], device, max_tokens=max_tokens))
            feats.append(_action_features_for_ref_cuda(env, refs[local], device))
            action_priors.append(_action_prior_for_ref_cuda(env, refs[local], mask_t, teacher, device))
            w, h = env.grid_shape()
            action_lengths.append(int(len(env.rotations) * w * h))

        if not valid_rows:
            continue

        tokens_batch, token_key_padding_mask = _left_pad_context_tokens(tokens, pad_value=0.0)
        action_feat_batch = _pad_2d_tensors(feats, pad_value=0.0)
        mask_batch = _pad_1d_tensors(masks, pad_value=0.0)
        bias_batch = _pad_1d_tensors(biases, pad_value=0.0)
        objective_teacher_batch = _pad_1d_tensors(objectives_teacher, pad_value=0.0)
        objective_metric_batch = _pad_1d_tensors(objectives_metric, pad_value=0.0)
        action_prior_batch = _pad_1d_tensors(action_priors, pad_value=0.0)

        policy_logits_batch, _rt, _sc, _side, _subzone, _pairwise, region_prior_batch, _enc = _policy_outputs_with_region_batch(
            model,
            valid_envs,
            valid_refs,
            tokens_batch,
            action_feat_batch,
            region_cfg,
            token_key_padding_mask=token_key_padding_mask,
        )
        rollout_score_batch, _rollout_candidates = score_actions_batched(
            policy_logits_batch,
            bias_batch,
            mask_batch,
            objective_teacher_batch,
            objective_metric_batch,
            region_prior_batch,
            action_prior_batch,
            teacher=teacher,
            objective_alpha=DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
            region_alpha=DEFAULT_ROLLOUT_REGION_ALPHA,
        )
        greedy_actions_t = torch.argmax(rollout_score_batch, dim=-1)
        if bool(sample_actions):
            temperature_value = max(1e-6, float(sampling_temperature))
            row_probs = torch.softmax(rollout_score_batch / temperature_value, dim=-1)
            row_sums = row_probs.sum(dim=-1, keepdim=True)
            valid_prob_rows = torch.isfinite(row_probs).all(dim=-1, keepdim=True) & (row_sums > 0.0)
            safe_probs = torch.where(
                valid_prob_rows,
                row_probs / row_sums.clamp_min(1.0e-12),
                torch.zeros_like(row_probs),
            )
            if not bool(valid_prob_rows.all().item()):
                # Preserve the old fallback semantics for numerically bad rows,
                # but do the sampling itself as a single batched GPU op.
                invalid_rows = (~valid_prob_rows).squeeze(1)
                safe_probs[invalid_rows] = 0.0
                safe_probs[invalid_rows, greedy_actions_t[invalid_rows]] = 1.0
            chosen_t = torch.multinomial(safe_probs, num_samples=1).squeeze(1)
        else:
            chosen_t = greedy_actions_t
        chosen = chosen_t.detach().cpu().tolist()
        for row, i in enumerate(valid_rows):
            a = int(chosen[row])
            A_i = int(action_lengths[row])
            if a < 0 or a >= A_i:
                envs[i].terminated = True
                terminated[i] = True
                if failure_reasons[i] is None:
                    failure_reasons[i] = "invalid_sampled_action"
                continue
            w, h, R = w_h_R[i]
            _obs, _r, _done, info = envs[i].step(
                _unflatten_action(a, w, h),
                assume_legal=True,
                return_observation=False,
                compute_objective=False,
            )
            if info.get("illegal"):
                envs[i].terminated = True
                terminated[i] = True
                if failure_reasons[i] is None:
                    failure_reasons[i] = f"illegal_action:{info.get('reason') or 'unknown'}"
                continue
            actions_flat[i].append(a)

    episodes: List[Dict[str, Any]] = []
    for ep_i, (path, env, acts) in enumerate(zip(paths, envs, actions_flat)):
        expected_dynamic_count = int(len(env.sequence))
        expected_count = int(env.total_expected_count() if hasattr(env, 'total_expected_count') else len(env.sequence))
        placed_count = int(len(env.placed))
        dynamic_placed_count = int(env.dynamic_placed_count() if hasattr(env, 'dynamic_placed_count') else placed_count)
        complete = bool(
            (acts or expected_dynamic_count == 0)
            and not bool(env.terminated)
            and dynamic_placed_count == expected_dynamic_count
            and placed_count == expected_count
            and int(env.t) >= expected_dynamic_count
        )
        partial_obj = _finite_partial_objective(env)
        failure_reason = None
        if not complete:
            if bool(env.terminated):
                failure_reason = failure_reasons[ep_i]
                if failure_reason is None:
                    failure_reason = "terminated_before_complete"
            elif placed_count != expected_count or dynamic_placed_count != expected_dynamic_count:
                failure_reason = "incomplete_layout"
            else:
                failure_reason = "no_valid_actions"

        failure_penalties: Dict[str, float] = {
            "missing_penalty": 0.0,
            "illegal_penalty": 0.0,
            "terminal_penalty": 0.0,
            "total_failure_penalty": 0.0,
            "missing_count": 0.0,
            "objective_scale_per_component": 0.0,
        }
        if complete:
            final_obj = float(partial_obj)
        else:
            final_obj, failure_penalties = _failed_rollout_objective_with_penalties(
                partial_obj=float(partial_obj),
                placed_count=placed_count,
                expected_count=expected_count,
                failure_reason=failure_reason,
                terminated=bool(env.terminated),
            )
        score = -float(final_obj)
        episodes.append({
            "task_path": path,
            "actions_flat": list(acts),
            "total_return": score,
            "final_obj": float(final_obj),
            "partial_obj": float(partial_obj),
            "failure_penalty": float(failure_penalties["total_failure_penalty"]),
            "failure_penalty_missing": float(failure_penalties["missing_penalty"]),
            "failure_penalty_illegal": float(failure_penalties["illegal_penalty"]),
            "failure_penalty_terminal": float(failure_penalties["terminal_penalty"]),
            "failure_missing_count": int(failure_penalties["missing_count"]),
            "score": score,
            "terminated": bool(env.terminated or not complete),
            "complete": bool(complete),
            "failure_reason": failure_reason,
            "placed_count": int(placed_count),
            "expected_count": int(expected_count),
            "dynamic_placed_count": int(dynamic_placed_count),
            "dynamic_expected_count": int(expected_dynamic_count),
            "steps": int(len(acts)),
            "sampled_policy": bool(sample_actions),
            "sampling_temperature": float(sampling_temperature),
        })
    return episodes


def replay_update_loss_on_episodes_batched(
    model: MaskedPolicy,
    episodes: List[Dict[str, Any]],
    k: int,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    expert_actions_by_path: Dict[str, List[Tuple[int, int, int]]],
    device: torch.device,
    action_ce_coef: float = 0.1,
    env_kwargs: Optional[Dict[str, Any]] = None,
    runtime_cache: Optional["_BoardRuntimeCache"] = None,
    *,
    sample_weights: Optional[List[float]] = None,
    advantages: Optional[List[float]] = None,
    distill_coef: float = 1.0,
    policy_gradient_coef: float = 0.0,
    entropy_coef: float = 0.0,
    rollout_temperature: float = 1.0,
    pg_policy_mode: str = "shaped",
    do_backward: bool = True,
    max_tokens: int = 128,
) -> torch.Tensor:
    """Batched replay teacher loss for several on-policy episodes.

    This is the replay counterpart of rollout_suffix_loss_batched_boards:
    multiple reconstructed episode states share one policy forward per depth.
    The action sequence still defines the environment state, but CPU objective
    recomputation is skipped while replaying and stepping.

    With ``policy_gradient_coef > 0``, the function also applies an on-policy
    REINFORCE term ``-advantage * log pi(a|s)`` and entropy regularization using
    the same objective-aware policy that generated replay rollouts.
    """
    if not episodes:
        return torch.tensor(0.0, device=device)
    _require_normalized_replay_episodes(
        episodes,
        context="replay update loss",
    )

    envs: List[PlacementEnv] = []
    paths: List[str] = []
    actions_all: List[List[int]] = []
    valid_episode = []
    for ep in episodes:
        path = str(ep["task_path"])
        actions = [int(a) for a in ep.get("actions_flat", [])]
        if not actions:
            valid_episode.append(False)
            envs.append(_fresh_env_for_path(path, env_kwargs, runtime_cache))
            paths.append(path)
            actions_all.append(actions)
            continue
        envs.append(_fresh_env_for_path(path, env_kwargs, runtime_cache))
        paths.append(path)
        actions_all.append(actions)
        valid_episode.append(True)

    B = len(envs)
    weights_in = sample_weights if sample_weights is not None else [1.0 / max(1, B)] * B
    weights_in = [float(w) for w in weights_in]
    if len(weights_in) != B:
        raise ValueError(
            f"sample_weights length mismatch: got {len(weights_in)}, expected {B}"
        )

    advantages_in = advantages if advantages is not None else [0.0] * B
    advantages_in = [float(value) for value in advantages_in]
    if len(advantages_in) != B:
        raise ValueError(
            f"advantages length mismatch: got {len(advantages_in)}, expected {B}"
        )
    pg_policy_mode = str(pg_policy_mode or "shaped").strip().lower()
    if pg_policy_mode not in {"shaped", "pure_logits"}:
        raise ValueError(
            f"pg_policy_mode must be 'shaped' or 'pure_logits'; got {pg_policy_mode!r}"
        )
    T_list = [len(env.sequence) for env in envs]
    prefix_lens = [max(0, int(T) - int(k)) for T in T_list]
    cur_t = [0 for _ in envs]
    active_ok = [bool(valid_episode[i]) for i in range(B)]
    w_h_R = [(env.grid_shape()[0], env.grid_shape()[1], len(env.rotations)) for env in envs]

    suffix_steps_target = []
    for i in range(B):
        max_suffix = max(0, min(T_list[i], len(actions_all[i])) - prefix_lens[i])
        suffix_steps_target.append(max(1, int(max_suffix)))

    # Reconstruct each on-policy prefix.  This is still per-episode because the
    # action sequence is different, but it skips observe() and CPU objective.
    for i, env in enumerate(envs):
        w, h, R = w_h_R[i]
        for t in range(min(prefix_lens[i], len(actions_all[i]))):
            a = int(actions_all[i][t])
            _obs, _r, _done, info = env.step(
                _unflatten_action(a, w, h),
                assume_legal=True,
                return_observation=False,
                compute_objective=False,
            )
            cur_t[i] += 1
            if info.get("illegal") or env.done():
                active_ok[i] = False
                break

    losses_for_log: List[torch.Tensor] = []
    loss_sum = 0.0
    loss_cnt = 0

    while True:
        candidate = [
            i for i in range(B)
            if active_ok[i]
            and cur_t[i] >= prefix_lens[i]
            and cur_t[i] < min(T_list[i], len(actions_all[i]))
            and (not envs[i].done())
        ]
        if not candidate:
            break

        candidate_envs = [envs[i] for i in candidate]
        refs = [env.current_ref() for env in candidate_envs]
        maps = _batch_objective_and_mask(candidate_envs, refs, device)

        valid_indices: List[int] = []
        valid_envs: List[PlacementEnv] = []
        valid_refs: List[str] = []
        valid_paths: List[str] = []
        masks: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []
        objectives_teacher: List[torch.Tensor] = []
        objectives_metric: List[torch.Tensor] = []
        tokens: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        action_priors: List[torch.Tensor] = []
        action_lengths: List[int] = []
        gt_indices: List[int] = []
        act_valid: List[float] = []
        expert_gt_actions: List[Optional[Tuple[int, int, int]]] = []

        for local, i in enumerate(candidate):
            env = envs[i]
            mask_map_t, bias_map_t, objective_teacher_map_t, objective_metric_map_t = maps[local]
            mask_t = mask_map_t.reshape(-1)
            bias_t = bias_map_t.reshape(-1)
            if not bool((mask_t > 0.5).any().item()):
                env.terminated = True
                active_ok[i] = False
                continue
            w, h, R = w_h_R[i]
            A = int(R * w * h)
            a_flat = int(actions_all[i][cur_t[i]])
            is_act_legal = 1.0 if (0 <= a_flat < A and bool((mask_t[a_flat] > 0.5).item())) else 0.0
            if a_flat < 0 or a_flat >= A:
                safe_gt = int(torch.where(mask_t > 0.5)[0][0].item())
            else:
                safe_gt = a_flat

            valid_indices.append(i)
            valid_envs.append(env)
            valid_refs.append(refs[local])
            valid_paths.append(paths[i])
            masks.append(mask_t)
            biases.append(bias_t)
            objectives_teacher.append(objective_teacher_map_t.reshape(-1))
            objectives_metric.append(objective_metric_map_t.reshape(-1))
            tokens.append(build_context_tokens_cuda(env, refs[local], device, max_tokens=max_tokens))
            feats.append(_action_features_for_ref_cuda(env, refs[local], device))
            action_priors.append(_action_prior_for_ref_cuda(env, refs[local], mask_t, teacher, device))
            action_lengths.append(A)
            gt_indices.append(int(safe_gt))
            act_valid.append(float(is_act_legal))
            expert_actions = expert_actions_by_path.get(paths[i], [])
            expert_gt_actions.append(expert_actions[cur_t[i]] if cur_t[i] < len(expert_actions) else None)

        if not valid_indices:
            continue

        tokens_batch, token_key_padding_mask = _left_pad_context_tokens(tokens, pad_value=0.0)
        action_feat_batch = _pad_2d_tensors(feats, pad_value=0.0)
        mask_batch = _pad_1d_tensors(masks, pad_value=0.0)
        bias_batch = _pad_1d_tensors(biases, pad_value=0.0)
        objective_teacher_batch = _pad_1d_tensors(objectives_teacher, pad_value=0.0)
        objective_metric_batch = _pad_1d_tensors(objectives_metric, pad_value=0.0)
        action_prior_batch = _pad_1d_tensors(action_priors, pad_value=0.0)

        policy_logits_batch, region_heatmap_logits, semantic_class_logits, side_preference_logits, subzone_logits, pairwise_logits, region_prior_batch, _enc = _policy_outputs_with_region_batch(
            model,
            valid_envs,
            valid_refs,
            tokens_batch,
            action_feat_batch,
            region_cfg,
            token_key_padding_mask=token_key_padding_mask,
        )
        logits_batch = (
            policy_logits_batch + bias_batch
        ).masked_fill(mask_batch < 0.5, -1e9)

        (
            loss_semantic, loss_region_heatmap, loss_prior_consistency, loss_side, loss_subzone, loss_pairwise,
            expert_weight, semantic_weight, teacher_weight, metric_weight, prior_supervision_weight,
        ) = _region_aux_losses_batch(
            refs=valid_refs,
            task_paths=valid_paths,
            envs=valid_envs,
            region_cfg=region_cfg,
            region_heatmap_logits=region_heatmap_logits,
            semantic_class_logits=semantic_class_logits,
            side_preference_logits=side_preference_logits,
            subzone_logits=subzone_logits,
            pairwise_logits=pairwise_logits,
            device=device,
        )
        expert_weight = torch.clamp(expert_weight, min=0.0, max=1.0)
        semantic_weight = torch.clamp(semantic_weight, min=0.0, max=1.0)
        teacher_weight = torch.clamp(teacher_weight, min=0.0, max=1.0)
        metric_weight = torch.clamp(metric_weight, min=0.0, max=1.0)
        prior_supervision_weight = torch.clamp(prior_supervision_weight, min=0.0, max=1.0)
        region_prior_for_teacher = None if region_prior_batch is None else region_prior_batch * teacher_weight.unsqueeze(1)

        q_batch, _cand = _build_teacher_distribution_batch_cuda(
            mask_batch,
            objective_teacher_batch,
            teacher,
            device=device,
            region_batch=region_prior_for_teacher,
            action_prior_batch=action_prior_batch,
        )
        logp_batch = torch.log_softmax(logits_batch, dim=-1)
        loss_teacher = -(q_batch * logp_batch).sum(dim=-1)
        probs_batch = torch.softmax(logits_batch, dim=-1)
        loss_metric = (probs_batch * objective_metric_batch).sum(dim=-1)

        gt_t = torch.tensor(gt_indices, device=device, dtype=torch.long)
        loss_act = F.cross_entropy(logits_batch, gt_t, reduction="none")
        loss_act = loss_act * torch.tensor(act_valid, device=device, dtype=loss_act.dtype)

        geometry_gt_indices = [
            None if a is None else _flatten_action(int(a[0]), int(a[1]), int(a[2]), *valid_envs[row].grid_shape())
            for row, a in enumerate(expert_gt_actions)
        ]
        loss_geo = _geometry_distill_loss_batch(
            envs=valid_envs,
            refs=valid_refs,
            logits_batch=logits_batch,
            action_feat_batch=action_feat_batch,
            geometry_gt_indices=geometry_gt_indices,
            geom_cfg=geom_cfg,
        )

        distill_loss = (
            teacher_weight * loss_teacher
            + float(action_ce_coef) * loss_act
            + metric_weight * float(teacher.metric_weight) * loss_metric
            + expert_weight * float(action_ce_coef) * loss_geo
            + expert_weight * float(region_cfg.aux_heatmap_weight) * loss_region_heatmap
            + prior_supervision_weight * float(region_cfg.aux_prior_consistency_weight) * loss_prior_consistency
            + semantic_weight * float(region_cfg.aux_semantic_weight) * loss_semantic
            + semantic_weight * float(region_cfg.aux_side_weight) * loss_side
            + semantic_weight * float(region_cfg.aux_subzone_weight) * loss_subzone
            + semantic_weight * float(region_cfg.aux_pairwise_weight) * loss_pairwise
        )

        if pg_policy_mode == "pure_logits":
            temp = max(1e-6, float(rollout_temperature))
            pure_scores = (policy_logits_batch / temp).masked_fill(mask_batch < 0.5, -1e9)
            policy_log_probs = torch.log_softmax(pure_scores, dim=-1)
            policy_probs = torch.softmax(pure_scores, dim=-1)
            policy_entropy = -(policy_probs * policy_log_probs).masked_fill(mask_batch < 0.5, 0.0).sum(dim=-1)
        else:
            policy_log_probs, policy_entropy, _policy_candidates = (
                policy_log_probs_batched(
                    policy_logits_batch,
                    bias_batch,
                    mask_batch,
                    objective_teacher_batch,
                    objective_metric_batch,
                    region_prior_batch,
                    action_prior_batch,
                    teacher=teacher,
                    temperature=float(rollout_temperature),
                    objective_alpha=DEFAULT_ROLLOUT_OBJECTIVE_ALPHA,
                    region_alpha=DEFAULT_ROLLOUT_REGION_ALPHA,
                )
            )
        selected_log_prob = policy_log_probs.gather(
            1,
            gt_t.unsqueeze(1),
        ).squeeze(1)
        action_valid_t = torch.tensor(
            act_valid,
            device=device,
            dtype=selected_log_prob.dtype,
        )
        advantage_t = torch.tensor(
            [advantages_in[i] for i in valid_indices],
            device=device,
            dtype=selected_log_prob.dtype,
        )
        policy_gradient_loss = (
            -advantage_t * selected_log_prob * action_valid_t
        )
        entropy_loss = -policy_entropy * action_valid_t

        loss_batch = (
            float(distill_coef) * distill_loss
            + float(policy_gradient_coef) * policy_gradient_loss
            + float(entropy_coef) * entropy_loss
        )

        scale = torch.tensor(
            [weights_in[i] / float(max(1, suffix_steps_target[i])) for i in valid_indices],
            device=device,
            dtype=loss_batch.dtype,
        )
        if do_backward:
            (loss_batch * scale).sum().backward()
        else:
            losses_for_log.append((loss_batch * scale).sum())

        # Log the same weighted objective that is used for backpropagation.
        # This avoids reporting a healthy unweighted loss when sample weights
        # are tiny or degenerate.
        loss_sum += float((loss_batch.detach() * scale).sum().item())

        for row, i in enumerate(valid_indices):
            a_flat = int(actions_all[i][cur_t[i]])
            w, h, R = w_h_R[i]
            _obs, _r, _done, info = envs[i].step(
                _unflatten_action(a_flat, w, h),
                assume_legal=True,
                return_observation=False,
                compute_objective=False,
            )
            cur_t[i] += 1
            if info.get("illegal"):
                active_ok[i] = False

    if do_backward:
        return torch.tensor(loss_sum, device=device)
    if not losses_for_log:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses_for_log).sum()



# Backward-compatible name retained for external callers.
replay_teacher_loss_on_episodes_batched = (
    replay_update_loss_on_episodes_batched
)


def _reset_board_k_stage(st: BoardCurriculumState, k_stage: int) -> None:
    st.k_stage = int(max(1, k_stage))
    st.k_stage_visits = 0
    st.k_stage_loss_ema = None
    st.k_stage_peak_ema = None
    st.k_stage_target_ema = None
    st.k_stage_target_locked = False
    st.k_stage_probe_left = 0
    st.k_stage_ready = False
    st.k_stage_deferred = False
    st.k_stage_defer_reason = None


def _ensure_board_k_stage(st: BoardCurriculumState, k_stage: int) -> None:
    if int(st.k_stage) != int(k_stage):
        _reset_board_k_stage(st, k_stage)


def _update_board_k_stage_state(
    st: BoardCurriculumState,
    *,
    k_stage: int,
    loss_val: float,
    ema_beta: float,
    probe_steps: int,
    drop_ratio: float,
    min_visits: int,
    max_visits: int = 0,
    max_visits_target_slack: float = 1.08,
    outlier_release_factor: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    _ensure_board_k_stage(st, k_stage)

    beta = min(max(float(ema_beta), 0.0), 0.999)
    if st.k_stage_loss_ema is None:
        st.k_stage_loss_ema = float(loss_val)
    else:
        st.k_stage_loss_ema = beta * float(st.k_stage_loss_ema) + (1.0 - beta) * float(loss_val)

    st.k_stage_visits += 1
    ema = float(st.k_stage_loss_ema)

    if st.k_stage_target_ema is not None:
        st.k_stage_target_locked = True

    if st.k_stage_probe_left <= 0 and (not st.k_stage_target_locked) and st.k_stage_target_ema is None:
        st.k_stage_probe_left = max(1, int(probe_steps))
        st.k_stage_peak_ema = ema

    if st.k_stage_probe_left > 0 and (not st.k_stage_target_locked):
        st.k_stage_peak_ema = ema if st.k_stage_peak_ema is None else max(float(st.k_stage_peak_ema), ema)
        st.k_stage_probe_left -= 1
        if st.k_stage_probe_left == 0 and st.k_stage_target_ema is None:
            st.k_stage_target_ema = float(st.k_stage_peak_ema) * (1.0 - _clip01(drop_ratio))
            st.k_stage_target_locked = True

    enough_visits = st.k_stage_visits >= max(1, int(min_visits))
    target_ready = st.k_stage_target_ema is not None and ema <= float(st.k_stage_target_ema)
    max_visits_hit = int(max_visits) > 0 and st.k_stage_visits >= int(max_visits)
    slack = max(1.0, float(max_visits_target_slack))
    forced_target_ready = (
        max_visits_hit
        and enough_visits
        and st.k_stage_target_ema is not None
        and ema <= float(st.k_stage_target_ema) * slack
    )
    outlier_release_hit = (
        int(max_visits) > 0
        and float(outlier_release_factor) > 1.0
        and st.k_stage_visits >= int(math.ceil(float(outlier_release_factor) * int(max_visits)))
    )

    if enough_visits and target_ready:
        st.k_stage_ready = True
        st.k_stage_deferred = False
        st.k_stage_defer_reason = None
        return True, 'stable'

    if forced_target_ready:
        st.k_stage_ready = True
        st.k_stage_deferred = False
        st.k_stage_defer_reason = None
        return True, 'max_visits_target'

    if outlier_release_hit and not st.k_stage_ready and not st.k_stage_deferred:
        st.k_stage_deferred = True
        st.k_stage_defer_reason = 'outlier_release'
        return False, 'outlier_deferred'

    if max_visits_hit:
        st.k_stage_ready = False
        return False, 'max_visits_hold'

    st.k_stage_ready = False
    return False, None


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 0.999)


def _recent_rel_change_small(vals: List[float], eps: float) -> bool:
    if len(vals) <= 1:
        return False
    eps = max(0.0, float(eps))
    rels: List[float] = []
    for prev, curr in zip(vals[:-1], vals[1:]):
        denom = max(abs(float(prev)), 1e-6)
        rels.append(abs(float(curr) - float(prev)) / denom)
    return bool(rels) and max(rels) <= eps


def _update_board_full_state(
    st: BoardCurriculumState,
    loss_val: float,
    main_step: int,
    ema_beta: float,
    probe_visits: int,
    drop_ratio: float,
    min_visits: int,
    plateau_patience: int,
    plateau_rel_change: float,
    max_visits: int = 0,
) -> Tuple[bool, Optional[str]]:
    beta = min(max(float(ema_beta), 0.0), 0.999)
    if st.full_loss_ema is None:
        st.full_loss_ema = float(loss_val)
    else:
        st.full_loss_ema = beta * float(st.full_loss_ema) + (1.0 - beta) * float(loss_val)

    st.full_visits += 1
    ema = float(st.full_loss_ema)

    if st.full_probe_left <= 0 and st.full_target_ema is None:
        st.full_probe_left = max(1, int(probe_visits))
        st.full_peak_ema = ema

    if st.full_probe_left > 0:
        st.full_peak_ema = ema if st.full_peak_ema is None else max(float(st.full_peak_ema), ema)
        st.full_probe_left -= 1
        if st.full_probe_left == 0:
            st.full_target_ema = float(st.full_peak_ema) * (1.0 - _clip01(drop_ratio))

    hist_keep = max(2, int(plateau_patience) + 1)
    st.recent_full_ema.append(ema)
    if len(st.recent_full_ema) > hist_keep:
        st.recent_full_ema = st.recent_full_ema[-hist_keep:]

    enough_visits = st.full_visits >= max(int(min_visits), int(probe_visits) + int(plateau_patience))
    target_ready = st.full_target_ema is not None and ema <= float(st.full_target_ema)
    plateau_ready = len(st.recent_full_ema) >= hist_keep and _recent_rel_change_small(
        st.recent_full_ema[-hist_keep:], float(plateau_rel_change)
    )

    if enough_visits and target_ready and plateau_ready:
        st.graduated = True
        st.graduate_step = int(main_step)
        st.graduate_reason = 'stable'
        return True, st.graduate_reason

    if int(max_visits) > 0 and st.full_visits >= int(max_visits):
        st.graduated = True
        st.graduate_step = int(main_step)
        st.graduate_reason = 'max_visits'
        return True, st.graduate_reason

    return False, None



def _board_state_to_dict(st: BoardCurriculumState) -> Dict[str, Any]:
    return {
        'index': int(st.index),
        'path': st.path,
        'kmax': int(st.kmax),
        'k_stage': int(st.k_stage),
        'k_stage_visits': int(st.k_stage_visits),
        'k_stage_loss_ema': None if st.k_stage_loss_ema is None else float(st.k_stage_loss_ema),
        'k_stage_peak_ema': None if st.k_stage_peak_ema is None else float(st.k_stage_peak_ema),
        'k_stage_target_ema': None if st.k_stage_target_ema is None else float(st.k_stage_target_ema),
        'k_stage_target_locked': bool(st.k_stage_target_locked),
        'k_stage_probe_left': int(st.k_stage_probe_left),
        'k_stage_ready': bool(st.k_stage_ready),
        'k_stage_deferred': bool(st.k_stage_deferred),
        'k_stage_defer_reason': st.k_stage_defer_reason,
        'graduated': bool(st.graduated),
        'full_visits': int(st.full_visits),
        'full_loss_ema': None if st.full_loss_ema is None else float(st.full_loss_ema),
        'full_peak_ema': None if st.full_peak_ema is None else float(st.full_peak_ema),
        'full_target_ema': None if st.full_target_ema is None else float(st.full_target_ema),
        'full_probe_left': int(st.full_probe_left),
        'recent_full_ema': [float(v) for v in st.recent_full_ema],
        'graduate_step': None if st.graduate_step is None else int(st.graduate_step),
        'graduate_reason': st.graduate_reason,
    }


def _board_state_from_dict(d: Dict[str, Any]) -> BoardCurriculumState:
    st = BoardCurriculumState(
        index=int(d['index']),
        path=str(d['path']),
        kmax=int(d['kmax']),
    )
    st.k_stage = int(d.get('k_stage', 1))
    st.k_stage_visits = int(d.get('k_stage_visits', 0))
    st.k_stage_loss_ema = None if d.get('k_stage_loss_ema') is None else float(d.get('k_stage_loss_ema'))
    st.k_stage_peak_ema = None if d.get('k_stage_peak_ema') is None else float(d.get('k_stage_peak_ema'))
    st.k_stage_target_ema = None if d.get('k_stage_target_ema') is None else float(d.get('k_stage_target_ema'))
    st.k_stage_target_locked = bool(d.get('k_stage_target_locked', st.k_stage_target_ema is not None))
    st.k_stage_probe_left = int(d.get('k_stage_probe_left', 0))
    st.k_stage_ready = bool(d.get('k_stage_ready', False))
    st.k_stage_deferred = bool(d.get('k_stage_deferred', False))
    st.k_stage_defer_reason = d.get('k_stage_defer_reason')
    st.graduated = bool(d.get('graduated', False))
    st.full_visits = int(d.get('full_visits', 0))
    st.full_loss_ema = None if d.get('full_loss_ema') is None else float(d.get('full_loss_ema'))
    st.full_peak_ema = None if d.get('full_peak_ema') is None else float(d.get('full_peak_ema'))
    st.full_target_ema = None if d.get('full_target_ema') is None else float(d.get('full_target_ema'))
    st.full_probe_left = int(d.get('full_probe_left', 0))
    st.recent_full_ema = [float(v) for v in d.get('recent_full_ema', [])]
    st.graduate_step = None if d.get('graduate_step') is None else int(d.get('graduate_step'))
    st.graduate_reason = d.get('graduate_reason')
    return st



def _normalize_board_state_path(path: str) -> str:
    """Return a stable local identity for matching resume board states."""
    expanded = os.path.expanduser(str(path))
    absolute = os.path.abspath(expanded)
    resolved = os.path.realpath(absolute)
    return os.path.normcase(os.path.normpath(resolved))


def _index_board_states_by_path(
    states: List[BoardCurriculumState],
    *,
    source_name: str,
) -> Dict[str, BoardCurriculumState]:
    indexed: Dict[str, BoardCurriculumState] = {}
    duplicate_details: List[str] = []
    for state in states:
        normalized = _normalize_board_state_path(state.path)
        if normalized in indexed:
            duplicate_details.append(
                f"{normalized!r} ({indexed[normalized].path!r}, {state.path!r})"
            )
            continue
        indexed[normalized] = state

    if duplicate_details:
        raise ValueError(
            f"Duplicate board paths in {source_name}: "
            + "; ".join(duplicate_details)
        )
    return indexed


def _restore_board_states_by_path(
    saved_states: List[BoardCurriculumState],
    current_states: List[BoardCurriculumState],
) -> List[BoardCurriculumState]:
    """Strictly restore curriculum states by normalized board path.

    Current task ordering is authoritative. Saved progress fields are copied
    onto that ordering, while current ``index``, ``path``, and ``kmax`` remain
    tied to the current task definitions.
    """
    saved_by_path = _index_board_states_by_path(
        saved_states,
        source_name="checkpoint board_states",
    )
    current_by_path = _index_board_states_by_path(
        current_states,
        source_name="current training tasks",
    )

    saved_paths = set(saved_by_path)
    current_paths = set(current_by_path)
    missing_from_checkpoint = sorted(current_paths - saved_paths)
    removed_from_current = sorted(saved_paths - current_paths)
    if missing_from_checkpoint or removed_from_current:
        details: List[str] = []
        if missing_from_checkpoint:
            details.append(
                "new/current-only="
                + repr(
                    [
                        current_by_path[path].path
                        for path in missing_from_checkpoint
                    ]
                )
            )
        if removed_from_current:
            details.append(
                "checkpoint-only="
                + repr(
                    [
                        saved_by_path[path].path
                        for path in removed_from_current
                    ]
                )
            )
        raise ValueError(
            "Resume board set does not match checkpoint board set: "
            + "; ".join(details)
        )

    restored: List[BoardCurriculumState] = []
    kmax_mismatches: List[str] = []
    for current_state in current_states:
        normalized = _normalize_board_state_path(current_state.path)
        saved_state = saved_by_path[normalized]
        if int(saved_state.kmax) != int(current_state.kmax):
            kmax_mismatches.append(
                f"{current_state.path!r}: "
                f"checkpoint={int(saved_state.kmax)}, "
                f"current={int(current_state.kmax)}"
            )
            continue

        rebound = copy.deepcopy(saved_state)
        rebound.index = int(current_state.index)
        rebound.path = str(current_state.path)
        rebound.kmax = int(current_state.kmax)
        restored.append(rebound)

    if kmax_mismatches:
        raise ValueError(
            "Resume board kmax mismatch: " + "; ".join(kmax_mismatches)
        )
    if len(restored) != len(current_states):
        raise RuntimeError(
            "Internal resume board-state restoration error: "
            f"restored={len(restored)}, current={len(current_states)}"
        )
    return restored



def _restore_replay_baselines_by_path(
    raw_baselines: Dict[str, Any],
    current_states: List[BoardCurriculumState],
) -> Dict[str, float]:
    """Rebind saved RL baselines to current task path spelling/order."""
    current_by_path = _index_board_states_by_path(
        current_states,
        source_name="current training tasks",
    )
    seen: Dict[str, str] = {}
    restored: Dict[str, float] = {}
    for saved_path, raw_value in raw_baselines.items():
        normalized = _normalize_board_state_path(str(saved_path))
        if normalized in seen:
            raise ValueError(
                "Duplicate normalized board paths in replay_baselines_by_path: "
                f"{seen[normalized]!r}, {saved_path!r}"
            )
        seen[normalized] = str(saved_path)
        if normalized not in current_by_path:
            raise ValueError(
                "Replay baseline checkpoint path is not present in current "
                f"training tasks: {saved_path!r}"
            )
        current_path = str(current_by_path[normalized].path)
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"Replay baseline must be finite; path={saved_path!r}, value={raw_value!r}"
            )
        restored[current_path] = value
    return restored


def _make_step_checkpoint_path(base_path: str, phase: str, step: int) -> str:
    p = Path(base_path)
    suffix = ''.join(p.suffixes)
    stem = p.name[:-len(suffix)] if suffix else p.name
    filename = f'{stem}.{phase}.step{int(step):07d}.pt'
    return str(p.with_name(filename))


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_name(path_obj.name + '.tmp')
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path_obj))


def _cleanup_old_checkpoints(base_path: str, keep_last: int) -> List[str]:
    if int(keep_last) <= 0:
        return []
    p = Path(base_path)
    stem = p.stem
    candidates = sorted(
        p.parent.glob(f'{stem}.*.step*.pt'),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for old in candidates[int(keep_last):]:
        try:
            old.unlink()
            removed.append(str(old))
        except FileNotFoundError:
            pass
    return removed


def _save_checkpoint(
    checkpoint_path: str,
    payload: Dict[str, Any],
    phase: str,
    step: int,
    keep_last: int,
) -> Tuple[str, str, List[str]]:
    latest_path = str(Path(checkpoint_path))
    step_path = _make_step_checkpoint_path(latest_path, phase=phase, step=int(step))
    _atomic_torch_save(payload, step_path)
    _atomic_torch_save(payload, latest_path)
    removed = _cleanup_old_checkpoints(latest_path, int(keep_last))
    return latest_path, step_path, removed


def _serialize_replay_buffer(rb: Optional[WeightedReplayBuffer]) -> Optional[Dict[str, Any]]:
    if rb is None:
        return None
    _require_normalized_replay_episodes(
        rb.items,
        context="replay checkpoint serialization",
    )
    if len(rb.priorities) != len(rb.items):
        raise ValueError(
            "Replay buffer priorities length mismatch before checkpoint: "
            f"items={len(rb.items)}, priorities={len(rb.priorities)}"
        )
    return {
        'capacity': int(rb.capacity),
        'alpha': float(rb.alpha),
        'score_kind': REPLAY_SCORE_KIND,
        'items': list(rb.items),
        'priorities': [float(v) for v in rb.priorities],
    }


def _restore_replay_buffer(data: Optional[Dict[str, Any]]) -> Optional[WeightedReplayBuffer]:
    if not data:
        return None
    rb = WeightedReplayBuffer(capacity=int(data.get('capacity', 2000)), alpha=float(data.get('alpha', 0.7)))
    raw_items = data.get('items', [])
    raw_priorities = data.get('priorities', [])
    if not isinstance(raw_items, list):
        raise ValueError("Replay checkpoint buffer items must be a list.")
    if not isinstance(raw_priorities, list):
        raise ValueError("Replay checkpoint buffer priorities must be a list.")
    if len(raw_priorities) != len(raw_items):
        raise ValueError(
            "Replay checkpoint buffer priorities length mismatch: "
            f"items={len(raw_items)}, priorities={len(raw_priorities)}"
        )
    rb.items = [dict(item) for item in raw_items]
    rb.priorities = [float(v) for v in raw_priorities]
    return rb


def _build_checkpoint_payload(
    *,
    model: MaskedPolicy,
    opt: AdamW,
    obs_dim: int,
    max_tokens: int,
    sequence_policy: str,
    board_states: List[BoardCurriculumState],
    kmax_final: int,
    warmup: int,
    steps: int,
    max_main_steps: int,
    adaptive_k: bool,
    fixed_k_steps: int,
    curriculum_window_mode: str,
    adaptive_k_drop_ratio: float,
    adaptive_k_drop_ratio_first: float,
    adaptive_k_probe_steps: int,
    adaptive_k_min_steps: int,
    adaptive_k_ema_beta: float,
    adaptive_k_max_steps_per_k: int,
    adaptive_k_max_visits_target_slack: float,
    adaptive_k_ready_ratio: float,
    adaptive_k_outlier_release_factor: float,
    board_full_ema_beta: float,
    board_full_probe_visits: int,
    board_full_drop_ratio: float,
    board_full_min_visits: int,
    board_full_plateau_patience: int,
    board_full_plateau_rel_change: float,
    board_full_max_visits: int,
    min_spacing_mm: float,
    env_alignment_bonus: float,
    env_edge_bonus: float,
    env_edge_eps_mm: float,
    reward_non_interface_edge_penalty: float,
    reward_non_interface_edge_margin_mm: float,
    reward_density_penalty: float,
    reward_density_radius_mm: float,
    reward_interior_penalty: float,
    reward_interior_margin_ratio: float,
    objective_hpwl_weight: float,
    objective_w_hpwl_weight: float,
    objective_nslw_weight: float,
    objective_region_weight: float,
    objective_module_region_weight: float,
    objective_module_floorplan_weight: float,
    module_region_bias: float,
    module_region_margin_mm: float,
    module_floorplan_separation_mm: float,
    module_floorplan_overlap_scale: float,
    module_floorplan_channel_scale: float,
    module_floorplan_compact_scale: float,
    module_floorplan_region_scale: float,
    objective_conn_weight: float,
    objective_align_weight: float,
    objective_group_weight: float,
    objective_anchor_weight: float,
    objective_boundary_group_weight: float,
    objective_pitch_weight: float,
    objective_orientation_weight: float,
    objective_edge_clearance_weight: float,
    objective_interior_weight: float,
    objective_density_weight: float,
    objective_soft_spacing_weight: float,
    objective_neatness_weight: float,
    edge_band_ratio: float,
    edge_band_center_ratio: float,
    soft_spacing_same_group_extra_mm: float,
    soft_spacing_cross_group_extra_mm: float,
    soft_spacing_large_extra_mm: float,
    same_group_density_scale: float,
    critical_neighbor_density_scale: float,
    anchor_group_density_scale: float,
    large_pair_density_scale: float,
    teacher: TeacherConfig,
    region_cfg: RegionPriorConfig,
    geom_cfg: GeometryDistillConfig,
    expert_mix_start: float,
    expert_mix_end: float,
    expert_mix_anneal_steps: int,
    replay_finetune: bool,
    replay_iters: int,
    replay_rollouts_per_iter: int,
    replay_update_steps: int,
    replay_batch_size: int,
    replay_capacity: int,
    replay_alpha: float,
    replay_temp: float,
    replay_action_ce_coef: float,
    replay_policy_gradient_coef: float,
    replay_entropy_coef: float,
    replay_rollout_temperature: float,
    replay_pg_policy_mode: str,
    replay_baseline_beta: float,
    replay_advantage_clip: float,
    replay_k: Optional[int],
    main_step: int,
    rr_ptr: int,
    step_cap_hit: bool,
    k_curr: int,
    loss_ema: Optional[float],
    k_steps: int,
    probe_left: int,
    peak_ema: Optional[float],
    target_ema: Optional[float],
    phase: str,
    replay_iter_done: int,
    replay_buffer: Optional[WeightedReplayBuffer],
    replay_best_score: Optional[float],
    replay_baselines_by_path: Dict[str, float],
) -> Dict[str, Any]:
    return {
        'format_version': 4,
        'phase': phase,
        'replay_iter_done': int(replay_iter_done),
        'replay_best_score': None if replay_best_score is None else float(replay_best_score),
        'replay_baselines_by_path': {
            str(path): float(value)
            for path, value in replay_baselines_by_path.items()
        },
        'replay_buffer': _serialize_replay_buffer(replay_buffer),
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'obs_dim': obs_dim,
        'model_cfg': dict(vars(model.cfg)),
        'action_feat_dim': int(ACTION_CONDITIONED_ACTION_FEAT_DIM),
        'region_grid_shape': [int(region_cfg.grid_x), int(region_cfg.grid_y)],
        'num_region_heatmap_bins': int(region_heatmap_num_bins(region_cfg.grid_x, region_cfg.grid_y)),
        'num_semantic_classes': int(len(SEMANTIC_CLASS_NAMES)),
        'num_side_preferences': int(len(SIDE_PREFERENCE_NAMES)),
        'num_subzones': int(len(SUBZONE_NAMES)),
        'num_pairwise_relations': int(len(PAIRWISE_RELATION_NAMES)),
        'max_tokens': int(max_tokens),
        'sequence_policy': str(sequence_policy),
        'action_scoring': {
            'version': ACTION_SCORING_VERSION,
            'objective_alpha': float(DEFAULT_ROLLOUT_OBJECTIVE_ALPHA),
            'region_alpha': float(DEFAULT_ROLLOUT_REGION_ALPHA),
        },
        'k_schedule': {
            'kmax_final': int(kmax_final),
            'warmup': int(warmup),
            'steps': int(steps),
            'max_main_steps': int(max_main_steps),
            'adaptive': bool(adaptive_k),
            'fixed_k_steps': int(fixed_k_steps),
            'curriculum_window_mode': str(curriculum_window_mode),
            'adaptive_drop_ratio': float(adaptive_k_drop_ratio),
            'adaptive_drop_ratio_first': float(adaptive_k_drop_ratio_first),
            'adaptive_probe_steps': int(adaptive_k_probe_steps),
            'adaptive_min_steps': int(adaptive_k_min_steps),
            'adaptive_ema_beta': float(adaptive_k_ema_beta),
            'adaptive_max_steps_per_k': int(adaptive_k_max_steps_per_k),
            'adaptive_max_visits_target_slack': float(adaptive_k_max_visits_target_slack),
            'adaptive_ready_ratio': float(adaptive_k_ready_ratio),
            'adaptive_outlier_release_factor': float(adaptive_k_outlier_release_factor),
        },
        'board_graduation': {
            'ema_beta': float(board_full_ema_beta),
            'probe_visits': int(board_full_probe_visits),
            'drop_ratio': float(board_full_drop_ratio),
            'min_visits': int(board_full_min_visits),
            'plateau_patience': int(board_full_plateau_patience),
            'plateau_rel_change': float(board_full_plateau_rel_change),
            'max_visits': int(board_full_max_visits),
        },
        'env_config': build_placement_env_kwargs(locals()),
        'board_states': [_board_state_to_dict(st) for st in board_states],
        'main_training': {
            'main_steps_executed': int(main_step),
            'step_cap_hit': bool(step_cap_hit),
            'all_boards_graduated': bool(all(st.graduated for st in board_states)),
            'rr_ptr': int(rr_ptr),
            'k_curr': int(k_curr),
            'loss_ema': None if loss_ema is None else float(loss_ema),
            'k_steps': int(k_steps),
            'probe_left': int(probe_left),
            'peak_ema': None if peak_ema is None else float(peak_ema),
            'target_ema': None if target_ema is None else float(target_ema),
            'board_k_tracking': {
                'per_board': True,
                'all_growth_boards_ready_required': True,
            },
        },
        'teacher': {
            'tau': float(teacher.tau),
            'lambda_region_prior': float(teacher.lambda_region_prior),
            'lambda_prior_region_heatmap': float(getattr(teacher, 'lambda_prior_region_heatmap', 0.10)),
            'lambda_conn_prior': float(teacher.lambda_conn_prior),
            'lambda_anchor_prior': float(teacher.lambda_anchor_prior),
            'lambda_module_prior': float(teacher.lambda_module_prior),
            'lambda_spacing_prior': float(teacher.lambda_spacing_prior),
            'lambda_edge_prior': float(teacher.lambda_edge_prior),
            'metric_weight': float(teacher.metric_weight),
            'topk': int(teacher.topk),
            'objective_delta_max': None if teacher.objective_delta_max is None else float(teacher.objective_delta_max),
            'gate_rollout': bool(teacher.gate_rollout),
        },
        'region_prior': {
            'enabled': bool(region_cfg.enabled),
            'grid_x': int(region_cfg.grid_x),
            'grid_y': int(region_cfg.grid_y),
            'heatmap_sigma_cells': float(region_cfg.heatmap_sigma_cells),
            'legacy_zone_edge_ratio': float(region_cfg.legacy_zone_edge_ratio),
            'legacy_zone_core_ratio': float(region_cfg.legacy_zone_core_ratio),
            'heatmap_action_prior_weight': float(region_cfg.heatmap_action_prior_weight),
            'aux_heatmap_weight': float(region_cfg.aux_heatmap_weight),
            'aux_prior_consistency_weight': float(region_cfg.aux_prior_consistency_weight),
            'aux_semantic_weight': float(region_cfg.aux_semantic_weight),
            'aux_side_weight': float(region_cfg.aux_side_weight),
            'aux_subzone_weight': float(region_cfg.aux_subzone_weight),
            'aux_pairwise_weight': float(region_cfg.aux_pairwise_weight),
            'region_type_names': list(REGION_TYPE_NAMES),
            'semantic_class_names': list(SEMANTIC_CLASS_NAMES),
        },
        'geometry_distill': {
            'xy_weight': float(geom_cfg.xy_weight),
            'rot_weight': float(geom_cfg.rot_weight),
            'align_offset_weight': float(geom_cfg.align_offset_weight),
            'boundary_axis_weight': float(geom_cfg.boundary_axis_weight),
            'module_bbox_weight': 0.0,
        },
        'expert_mix': {
            'start': float(expert_mix_start),
            'end': float(expert_mix_end),
            'anneal_steps': int(expert_mix_anneal_steps),
        },
        'replay': {
            'enabled': bool(replay_finetune),
            'score_kind': REPLAY_SCORE_KIND,
            'score_contract': 'all replay episode scores are relative expert improvement for the same board',
            'iters': int(replay_iters),
            'rollouts_per_iter': int(replay_rollouts_per_iter),
            'update_steps': int(replay_update_steps),
            'batch_size': int(replay_batch_size),
            'capacity': int(replay_capacity),
            'alpha': float(replay_alpha),
            'temp': float(replay_temp),
            'action_ce_coef': float(replay_action_ce_coef),
            'policy_gradient_coef': float(replay_policy_gradient_coef),
            'entropy_coef': float(replay_entropy_coef),
            'rollout_temperature': float(replay_rollout_temperature),
            'pg_policy_mode': str(replay_pg_policy_mode),
            'baseline_beta': float(replay_baseline_beta),
            'advantage_clip': float(replay_advantage_clip),
            'rl_algorithm': 'reinforce_per_board_ema_v1',
            'k': None if replay_k is None else int(replay_k),
        },
    }


def train(
    train_tasks: List[Dict[str, Any]],
    steps: int = 0,
    warmup: int = 1500,
    kmax_final: int = -1,
    # batch training
    batch_size: int = 1,
    # k curriculum
    adaptive_k: bool = False,
    fixed_k_steps: int = 0,
    curriculum_window_mode: str = 'suffix',
    adaptive_k_drop_ratio: float = 0.7,
    adaptive_k_drop_ratio_first: float = 0.4,
    adaptive_k_probe_steps: int = 200,
    adaptive_k_min_steps: int = 400,
    adaptive_k_ema_beta: float = 0.98,
    adaptive_k_max_steps_per_k: int = 8000,
    adaptive_k_max_visits_target_slack: float = 1.08,
    adaptive_k_ready_ratio: float = 0.91,
    adaptive_k_outlier_release_factor: float = 3.0,
    max_main_steps: int = 0,
    # per-board graduation after full-board training
    board_full_ema_beta: float = 0.90,
    board_full_probe_visits: int = 3,
    board_full_drop_ratio: float = 0.20,
    board_full_min_visits: int = 5,
    board_full_plateau_patience: int = 3,
    board_full_plateau_rel_change: float = 0.02,
    board_full_max_visits: int = 0,
    lr: Optional[float] = None,
    seed: int = 7,
    save_path: str = 'model.pt',
    device: str = 'cuda',
    max_tokens: int = 128,
    sequence_policy: str = 'rebuild',
    model_d_model: int = 256,
    model_nhead: int = 8,
    model_num_layers: int = 6,
    model_dropout: float = 0.1,
    min_spacing_mm: float = 0.2,
    env_alignment_bonus: float = 0.05,
    env_edge_bonus: float = 0.15,
    env_edge_eps_mm: float = 1.5,
    reward_non_interface_edge_penalty: float = 10.0,
    reward_non_interface_edge_margin_mm: float = 2.5,
    reward_density_penalty: float = 3.0,
    reward_density_radius_mm: float = 4.0,
    reward_interior_penalty: float = 1.0,
    reward_interior_margin_ratio: float = 0.18,

    # 1) soft teacher q(a)
    teacher_tau: float = 0.5,
    teacher_lambda_region_prior: float = 0.15,
    teacher_lambda_prior_region_heatmap: float = 0.10,
    teacher_lambda_conn_prior: float = 0.15,
    teacher_lambda_anchor_prior: float = 0.20,
    teacher_lambda_module_prior: float = 0.15,
    teacher_lambda_spacing_prior: float = 0.08,
    teacher_lambda_edge_prior: float = 0.05,
    teacher_metric_weight: float = 0.15,
    objective_hpwl_weight: float = 1.0,
    objective_w_hpwl_weight: float = 0.2,
    objective_nslw_weight: float = 0.05,
    objective_region_weight: float = 0.55,
    objective_module_region_weight: float = 0.35,
    objective_module_floorplan_weight: float = 0.0,
    module_region_bias: float = 0.20,
    module_region_margin_mm: float = 2.0,
    module_floorplan_separation_mm: float = 2.0,
    module_floorplan_overlap_scale: float = 1.0,
    module_floorplan_channel_scale: float = 0.65,
    module_floorplan_compact_scale: float = 0.20,
    module_floorplan_region_scale: float = 0.60,
    objective_conn_weight: float = 0.50,
    objective_align_weight: float = 0.28,
    objective_group_weight: float = 0.12,
    objective_anchor_weight: float = 0.18,
    objective_boundary_group_weight: float = 0.22,
    objective_pitch_weight: float = 0.22,
    objective_orientation_weight: float = 0.14,
    objective_edge_clearance_weight: float = 0.40,
    objective_interior_weight: float = 0.30,
    objective_density_weight: float = 0.45,
    objective_soft_spacing_weight: float = 0.32,
    objective_neatness_weight: float = 0.12,
    edge_band_ratio: float = 0.12,
    edge_band_center_ratio: float = 0.55,
    soft_spacing_same_group_extra_mm: float = 0.6,
    soft_spacing_cross_group_extra_mm: float = 1.4,
    soft_spacing_large_extra_mm: float = 0.7,
    same_group_density_scale: float = 0.40,
    critical_neighbor_density_scale: float = 0.25,
    anchor_group_density_scale: float = 0.50,
    large_pair_density_scale: float = 1.20,

    region_prior_enabled: bool = True,
    region_grid_x: int = 6,
    region_grid_y: int = 6,
    region_heatmap_sigma_cells: float = 0.85,
    legacy_region_zone_edge_ratio: float = 0.12,
    legacy_region_zone_core_ratio: float = 0.28,
    region_heatmap_action_prior_weight: float = 0.35,
    region_aux_heatmap_weight: float = 0.30,
    region_aux_prior_consistency_weight: float = 0.03,
    region_aux_semantic_weight: float = 0.10,
    region_aux_side_weight: float = 0.12,
    region_aux_subzone_weight: float = 0.10,
    region_aux_pairwise_weight: float = 0.10,

    geometry_xy_weight: float = 0.05,
    geometry_rot_weight: float = 0.02,
    geometry_align_offset_weight: float = 0.03,
    geometry_boundary_axis_weight: float = 0.03,
    geometry_module_bbox_weight: float = 0.0,

    # 3) Top-K / threshold teacher (objective-guided)
    teacher_topk: int = 256,
    teacher_objective_delta_max: Optional[float] = None,
    teacher_gate_rollout: bool = True,

    # 2) expert mix annealing
    expert_mix_start: float = 1.0,
    expert_mix_end: float = 0.40,
    expert_mix_anneal_steps: int = 80000,

    # 4) on-policy REINFORCE + weighted replay self-imitation
    replay_finetune: bool = True,
    replay_iters: int = 20,
    replay_rollouts_per_iter: int = 16,
    replay_update_steps: int = 32,
    replay_batch_size: int = 4,
    replay_capacity: int = 2000,
    replay_alpha: float = 0.7,
    replay_temp: float = 1.0,
    replay_action_ce_coef: float = 0.1,
    replay_policy_gradient_coef: float = 1.0,
    replay_entropy_coef: float = 0.01,
    replay_rollout_temperature: float = 1.0,
    replay_pg_policy_mode: str = "shaped",
    replay_baseline_beta: float = 0.90,
    replay_advantage_clip: float = 1.0,
    replay_k: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every_steps: int = 0,
    keep_last_checkpoints: int = 0,
    resume: bool = False,
    resume_restore_config: bool = True,
    visdom_enabled: bool = False,
    visdom_server: str = 'http://localhost',
    visdom_port: int = 8097,
    visdom_env: str = 'pcb_autoplace',
    visdom_prefix: str = 'train',
    visdom_log_interval: int = 200,
):
    """Training with fused masked learning.

    Main phase:
      - Keep a global k curriculum (adaptive or scheduled).
      - Every board uses k_eff = min(global_k, board_kmax).
      - Once a board reaches full-board training (k_eff == board_kmax), it no longer waits for a fixed
        number of visits. Instead, it maintains its own EMA loss, target EMA, and plateau history.
      - When that board's full-board loss both drops enough and plateaus, the board graduates and is
        removed from the main training pool.

    Replay phase:
      - After main training, collect stochastic on-policy trajectories for REINFORCE, then run weighted replay self-imitation updates.
    """
    if not train_tasks:
        raise ValueError('train_tasks is empty.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_t = torch.device(device)
    if device_t.type != 'cuda':
        raise ValueError(f'train() is CUDA-only; got device={device!r}')
    if not torch.cuda.is_available():
        raise RuntimeError('train() requires CUDA, but torch.cuda.is_available() is False.')

    # Load resume metadata before constructing any model/config/environment
    # objects.  This is essential: restoring after construction silently mixes
    # checkpoint weights with the caller's current/default training settings.
    explicit_resume_lr = None if lr is None else float(lr)
    resume_ckpt: Optional[Dict[str, Any]] = None
    resume_values: Dict[str, Any] = {}
    if bool(resume):
        if not checkpoint_path:
            raise ValueError('resume=True requires checkpoint_path to be set.')
        resume_ckpt = torch.load(checkpoint_path, map_location=device_t)
        if not isinstance(resume_ckpt, dict):
            raise ValueError(f'Resume checkpoint must contain a dict payload: {checkpoint_path}')
        if 'model_state' not in resume_ckpt:
            raise ValueError(f'Resume checkpoint is missing model_state: {checkpoint_path}')
        if 'optimizer_state' not in resume_ckpt:
            raise ValueError(f'Resume checkpoint is missing optimizer_state: {checkpoint_path}')
        if bool(resume_restore_config):
            resume_values = _resume_train_config_from_checkpoint(resume_ckpt)
            print(
                f'[resume] restoring {len(resume_values)} saved configuration values '
                'before model/environment construction'
            )
        else:
            print('[resume] saved configuration restoration disabled; using current train() arguments')

    def _resume_value(name: str, current: Any) -> Any:
        return resume_values.get(name, current)

    # Schedule and curriculum configuration.
    steps = int(_resume_value('steps', steps))
    warmup = int(_resume_value('warmup', warmup))
    kmax_final = int(_resume_value('kmax_final', kmax_final))
    adaptive_k = bool(_resume_value('adaptive_k', adaptive_k))
    fixed_k_steps = int(_resume_value('fixed_k_steps', fixed_k_steps))
    curriculum_window_mode = str(_resume_value('curriculum_window_mode', curriculum_window_mode)).strip().lower()
    if curriculum_window_mode not in {'suffix', 'prefix', 'random', 'mixed'}:
        raise ValueError(
            f"curriculum_window_mode must be suffix/prefix/random/mixed, got {curriculum_window_mode!r}"
        )
    adaptive_k_drop_ratio = float(_resume_value('adaptive_k_drop_ratio', adaptive_k_drop_ratio))
    adaptive_k_drop_ratio_first = float(_resume_value('adaptive_k_drop_ratio_first', adaptive_k_drop_ratio_first))
    adaptive_k_probe_steps = int(_resume_value('adaptive_k_probe_steps', adaptive_k_probe_steps))
    adaptive_k_min_steps = int(_resume_value('adaptive_k_min_steps', adaptive_k_min_steps))
    adaptive_k_ema_beta = float(_resume_value('adaptive_k_ema_beta', adaptive_k_ema_beta))
    adaptive_k_max_steps_per_k = int(_resume_value('adaptive_k_max_steps_per_k', adaptive_k_max_steps_per_k))
    adaptive_k_max_visits_target_slack = float(_resume_value('adaptive_k_max_visits_target_slack', adaptive_k_max_visits_target_slack))
    adaptive_k_ready_ratio = float(_resume_value('adaptive_k_ready_ratio', adaptive_k_ready_ratio))
    adaptive_k_outlier_release_factor = float(_resume_value('adaptive_k_outlier_release_factor', adaptive_k_outlier_release_factor))
    max_main_steps = int(_resume_value('max_main_steps', max_main_steps))

    board_full_ema_beta = float(_resume_value('board_full_ema_beta', board_full_ema_beta))
    board_full_probe_visits = int(_resume_value('board_full_probe_visits', board_full_probe_visits))
    board_full_drop_ratio = float(_resume_value('board_full_drop_ratio', board_full_drop_ratio))
    board_full_min_visits = int(_resume_value('board_full_min_visits', board_full_min_visits))
    board_full_plateau_patience = int(_resume_value('board_full_plateau_patience', board_full_plateau_patience))
    board_full_plateau_rel_change = float(_resume_value('board_full_plateau_rel_change', board_full_plateau_rel_change))
    board_full_max_visits = int(_resume_value('board_full_max_visits', board_full_max_visits))

    # Model-input and auxiliary-loss configuration.
    max_tokens = int(_resume_value('max_tokens', max_tokens))
    sequence_policy = normalize_sequence_policy(
        _resume_value('sequence_policy', sequence_policy)
    )
    teacher_tau = float(_resume_value('teacher_tau', teacher_tau))
    teacher_lambda_region_prior = float(_resume_value('teacher_lambda_region_prior', teacher_lambda_region_prior))
    teacher_lambda_prior_region_heatmap = float(_resume_value('teacher_lambda_prior_region_heatmap', teacher_lambda_prior_region_heatmap))
    teacher_lambda_conn_prior = float(_resume_value('teacher_lambda_conn_prior', teacher_lambda_conn_prior))
    teacher_lambda_anchor_prior = float(_resume_value('teacher_lambda_anchor_prior', teacher_lambda_anchor_prior))
    teacher_lambda_module_prior = float(_resume_value('teacher_lambda_module_prior', teacher_lambda_module_prior))
    teacher_lambda_spacing_prior = float(_resume_value('teacher_lambda_spacing_prior', teacher_lambda_spacing_prior))
    teacher_lambda_edge_prior = float(_resume_value('teacher_lambda_edge_prior', teacher_lambda_edge_prior))
    teacher_metric_weight = float(_resume_value('teacher_metric_weight', teacher_metric_weight))
    teacher_topk = int(_resume_value('teacher_topk', teacher_topk))
    teacher_objective_delta_max = _resume_value('teacher_objective_delta_max', teacher_objective_delta_max)
    if teacher_objective_delta_max is not None:
        teacher_objective_delta_max = float(teacher_objective_delta_max)
    teacher_gate_rollout = bool(_resume_value('teacher_gate_rollout', teacher_gate_rollout))

    region_prior_enabled = bool(_resume_value('region_prior_enabled', region_prior_enabled))
    region_grid_x = int(_resume_value('region_grid_x', region_grid_x))
    region_grid_y = int(_resume_value('region_grid_y', region_grid_y))
    region_heatmap_sigma_cells = float(_resume_value('region_heatmap_sigma_cells', region_heatmap_sigma_cells))
    legacy_region_zone_edge_ratio = float(_resume_value('legacy_region_zone_edge_ratio', legacy_region_zone_edge_ratio))
    legacy_region_zone_core_ratio = float(_resume_value('legacy_region_zone_core_ratio', legacy_region_zone_core_ratio))
    region_heatmap_action_prior_weight = float(_resume_value('region_heatmap_action_prior_weight', region_heatmap_action_prior_weight))
    region_aux_heatmap_weight = float(_resume_value('region_aux_heatmap_weight', region_aux_heatmap_weight))
    region_aux_prior_consistency_weight = float(_resume_value('region_aux_prior_consistency_weight', region_aux_prior_consistency_weight))
    region_aux_semantic_weight = float(_resume_value('region_aux_semantic_weight', region_aux_semantic_weight))
    region_aux_side_weight = float(_resume_value('region_aux_side_weight', region_aux_side_weight))
    region_aux_subzone_weight = float(_resume_value('region_aux_subzone_weight', region_aux_subzone_weight))
    region_aux_pairwise_weight = float(_resume_value('region_aux_pairwise_weight', region_aux_pairwise_weight))

    geometry_xy_weight = float(_resume_value('geometry_xy_weight', geometry_xy_weight))
    geometry_rot_weight = float(_resume_value('geometry_rot_weight', geometry_rot_weight))
    geometry_align_offset_weight = float(_resume_value('geometry_align_offset_weight', geometry_align_offset_weight))
    geometry_boundary_axis_weight = float(_resume_value('geometry_boundary_axis_weight', geometry_boundary_axis_weight))
    geometry_module_bbox_weight = float(_resume_value('geometry_module_bbox_weight', geometry_module_bbox_weight))

    expert_mix_start = float(_resume_value('expert_mix_start', expert_mix_start))
    expert_mix_end = float(_resume_value('expert_mix_end', expert_mix_end))
    expert_mix_anneal_steps = int(_resume_value('expert_mix_anneal_steps', expert_mix_anneal_steps))

    replay_finetune = bool(_resume_value('replay_finetune', replay_finetune))
    replay_iters = int(_resume_value('replay_iters', replay_iters))
    replay_rollouts_per_iter = int(_resume_value('replay_rollouts_per_iter', replay_rollouts_per_iter))
    replay_update_steps = int(_resume_value('replay_update_steps', replay_update_steps))
    replay_batch_size = int(_resume_value('replay_batch_size', replay_batch_size))
    replay_capacity = int(_resume_value('replay_capacity', replay_capacity))
    replay_alpha = float(_resume_value('replay_alpha', replay_alpha))
    replay_temp = float(_resume_value('replay_temp', replay_temp))
    replay_action_ce_coef = float(_resume_value('replay_action_ce_coef', replay_action_ce_coef))
    replay_policy_gradient_coef = float(
        _resume_value(
            'replay_policy_gradient_coef',
            replay_policy_gradient_coef,
        )
    )
    replay_entropy_coef = float(
        _resume_value('replay_entropy_coef', replay_entropy_coef)
    )
    replay_rollout_temperature = max(
        1e-6,
        float(
            _resume_value(
                'replay_rollout_temperature',
                replay_rollout_temperature,
            )
        ),
    )
    replay_pg_policy_mode = str(
        _resume_value('replay_pg_policy_mode', replay_pg_policy_mode)
    ).strip().lower()
    if replay_pg_policy_mode not in {"shaped", "pure_logits"}:
        raise ValueError(
            f"replay_pg_policy_mode must be 'shaped' or 'pure_logits'; got {replay_pg_policy_mode!r}"
        )
    replay_baseline_beta = float(
        min(
            0.999999,
            max(
                0.0,
                _resume_value(
                    'replay_baseline_beta',
                    replay_baseline_beta,
                ),
            ),
        )
    )
    replay_advantage_clip = max(
        0.0,
        float(
            _resume_value(
                'replay_advantage_clip',
                replay_advantage_clip,
            )
        ),
    )
    if replay_policy_gradient_coef < 0.0:
        raise ValueError(
            f"replay_policy_gradient_coef must be >= 0; got {replay_policy_gradient_coef}"
        )
    if replay_entropy_coef < 0.0:
        raise ValueError(
            f"replay_entropy_coef must be >= 0; got {replay_entropy_coef}"
        )
    replay_k = _resume_value('replay_k', replay_k)
    if replay_k is not None:
        replay_k = int(replay_k)

    # PlacementEnv objective/mask configuration.
    min_spacing_mm = float(_resume_value('min_spacing_mm', min_spacing_mm))
    env_alignment_bonus = float(_resume_value('env_alignment_bonus', env_alignment_bonus))
    env_edge_bonus = float(_resume_value('env_edge_bonus', env_edge_bonus))
    env_edge_eps_mm = float(_resume_value('env_edge_eps_mm', env_edge_eps_mm))
    reward_non_interface_edge_penalty = float(_resume_value('reward_non_interface_edge_penalty', reward_non_interface_edge_penalty))
    reward_non_interface_edge_margin_mm = float(_resume_value('reward_non_interface_edge_margin_mm', reward_non_interface_edge_margin_mm))
    reward_density_penalty = float(_resume_value('reward_density_penalty', reward_density_penalty))
    reward_density_radius_mm = float(_resume_value('reward_density_radius_mm', reward_density_radius_mm))
    reward_interior_penalty = float(_resume_value('reward_interior_penalty', reward_interior_penalty))
    reward_interior_margin_ratio = float(_resume_value('reward_interior_margin_ratio', reward_interior_margin_ratio))
    objective_hpwl_weight = float(_resume_value('objective_hpwl_weight', objective_hpwl_weight))
    objective_w_hpwl_weight = float(_resume_value('objective_w_hpwl_weight', objective_w_hpwl_weight))
    objective_nslw_weight = float(_resume_value('objective_nslw_weight', objective_nslw_weight))
    objective_region_weight = float(_resume_value('objective_region_weight', objective_region_weight))
    objective_module_region_weight = float(_resume_value('objective_module_region_weight', objective_module_region_weight))
    objective_module_floorplan_weight = float(_resume_value('objective_module_floorplan_weight', objective_module_floorplan_weight))
    module_region_bias = float(_resume_value('module_region_bias', module_region_bias))
    module_region_margin_mm = float(_resume_value('module_region_margin_mm', module_region_margin_mm))
    module_floorplan_separation_mm = float(_resume_value('module_floorplan_separation_mm', module_floorplan_separation_mm))
    module_floorplan_overlap_scale = float(_resume_value('module_floorplan_overlap_scale', module_floorplan_overlap_scale))
    module_floorplan_channel_scale = float(_resume_value('module_floorplan_channel_scale', module_floorplan_channel_scale))
    module_floorplan_compact_scale = float(_resume_value('module_floorplan_compact_scale', module_floorplan_compact_scale))
    module_floorplan_region_scale = float(_resume_value('module_floorplan_region_scale', module_floorplan_region_scale))
    objective_conn_weight = float(_resume_value('objective_conn_weight', objective_conn_weight))
    objective_align_weight = float(_resume_value('objective_align_weight', objective_align_weight))
    objective_group_weight = float(_resume_value('objective_group_weight', objective_group_weight))
    objective_anchor_weight = float(_resume_value('objective_anchor_weight', objective_anchor_weight))
    objective_boundary_group_weight = float(_resume_value('objective_boundary_group_weight', objective_boundary_group_weight))
    objective_pitch_weight = float(_resume_value('objective_pitch_weight', objective_pitch_weight))
    objective_orientation_weight = float(_resume_value('objective_orientation_weight', objective_orientation_weight))
    objective_edge_clearance_weight = float(_resume_value('objective_edge_clearance_weight', objective_edge_clearance_weight))
    objective_interior_weight = float(_resume_value('objective_interior_weight', objective_interior_weight))
    objective_density_weight = float(_resume_value('objective_density_weight', objective_density_weight))
    objective_soft_spacing_weight = float(_resume_value('objective_soft_spacing_weight', objective_soft_spacing_weight))
    objective_neatness_weight = float(_resume_value('objective_neatness_weight', objective_neatness_weight))
    edge_band_ratio = float(_resume_value('edge_band_ratio', edge_band_ratio))
    edge_band_center_ratio = float(_resume_value('edge_band_center_ratio', edge_band_center_ratio))
    soft_spacing_same_group_extra_mm = float(_resume_value('soft_spacing_same_group_extra_mm', soft_spacing_same_group_extra_mm))
    soft_spacing_cross_group_extra_mm = float(_resume_value('soft_spacing_cross_group_extra_mm', soft_spacing_cross_group_extra_mm))
    soft_spacing_large_extra_mm = float(_resume_value('soft_spacing_large_extra_mm', soft_spacing_large_extra_mm))
    same_group_density_scale = float(_resume_value('same_group_density_scale', same_group_density_scale))
    critical_neighbor_density_scale = float(_resume_value('critical_neighbor_density_scale', critical_neighbor_density_scale))
    anchor_group_density_scale = float(_resume_value('anchor_group_density_scale', anchor_group_density_scale))
    large_pair_density_scale = float(_resume_value('large_pair_density_scale', large_pair_density_scale))

    checkpoint_lrs = _optimizer_learning_rates_from_checkpoint(resume_ckpt or {})
    if explicit_resume_lr is not None:
        lr = float(explicit_resume_lr)
    elif checkpoint_lrs:
        lr = float(checkpoint_lrs[0])
    else:
        lr = float(DEFAULT_LEARNING_RATE)

    Tmax = max(len(it['expert_actions']) for it in train_tasks)
    if int(kmax_final) <= 0 or int(kmax_final) < Tmax:
        if int(kmax_final) > 0 and int(kmax_final) < Tmax:
            print(
                f'[warn] kmax_final={int(kmax_final)} < Tmax={Tmax}; '
                'raise to Tmax so every board can reach full-board and graduate.'
            )
        kmax_final = Tmax
        print(f'[info] full masking enabled: kmax_final={kmax_final} (Tmax={Tmax})')

    obs_dim = OBS_TOKEN_DIM
    requested_model_cfg = _validate_model_config(
        ModelConfig(
            d_model=int(model_d_model),
            nhead=int(model_nhead),
            num_layers=int(model_num_layers),
            dropout=float(model_dropout),
        )
    )
    model_cfg = _model_config_from_checkpoint(resume_ckpt, fallback=requested_model_cfg)
    region_cfg = RegionPriorConfig(
        enabled=bool(region_prior_enabled),
        grid_x=int(region_grid_x),
        grid_y=int(region_grid_y),
        heatmap_sigma_cells=float(region_heatmap_sigma_cells),
        legacy_zone_edge_ratio=float(legacy_region_zone_edge_ratio),
        legacy_zone_core_ratio=float(legacy_region_zone_core_ratio),
        heatmap_action_prior_weight=float(region_heatmap_action_prior_weight),
        aux_heatmap_weight=float(region_aux_heatmap_weight),
        aux_prior_consistency_weight=float(region_aux_prior_consistency_weight),
        aux_semantic_weight=float(region_aux_semantic_weight),
        aux_side_weight=float(region_aux_side_weight),
        aux_subzone_weight=float(region_aux_subzone_weight),
        aux_pairwise_weight=float(region_aux_pairwise_weight),
    )
    _validate_region_heatmap_grid_for_tasks(train_tasks, region_cfg)
    if resume_ckpt is not None:
        _validate_resume_model_metadata(resume_ckpt, region_cfg)

    model = MaskedPolicy(
        obs_dim=obs_dim,
        cfg=model_cfg,
        action_feat_dim=ACTION_CONDITIONED_ACTION_FEAT_DIM,
        region_grid_shape=(int(region_cfg.grid_x), int(region_cfg.grid_y)),
        num_region_heatmap_bins=int(region_heatmap_num_bins(region_cfg.grid_x, region_cfg.grid_y)),
        num_semantic_classes=int(len(SEMANTIC_CLASS_NAMES)),
        num_side_preferences=int(len(SIDE_PREFERENCE_NAMES)),
        num_subzones=int(len(SUBZONE_NAMES)),
        num_pairwise_relations=int(len(PAIRWISE_RELATION_NAMES)),
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr)

    teacher = TeacherConfig(
        tau=float(teacher_tau),
        lambda_region_prior=float(teacher_lambda_region_prior),
        lambda_prior_region_heatmap=float(teacher_lambda_prior_region_heatmap),
        lambda_conn_prior=float(teacher_lambda_conn_prior),
        lambda_anchor_prior=float(teacher_lambda_anchor_prior),
        lambda_module_prior=float(teacher_lambda_module_prior),
        lambda_spacing_prior=float(teacher_lambda_spacing_prior),
        lambda_edge_prior=float(teacher_lambda_edge_prior),
        metric_weight=float(teacher_metric_weight),
        topk=int(teacher_topk),
        objective_delta_max=None if teacher_objective_delta_max is None else float(teacher_objective_delta_max),
        gate_rollout=bool(teacher_gate_rollout),
    )
    geom_cfg = GeometryDistillConfig(
        xy_weight=float(geometry_xy_weight),
        rot_weight=float(geometry_rot_weight),
        align_offset_weight=float(geometry_align_offset_weight),
        boundary_axis_weight=float(geometry_boundary_axis_weight),
        module_bbox_weight=float(geometry_module_bbox_weight),
    )
    expert_actions_by_path: Dict[str, List[Tuple[int, int, int]]] = {
        str(it['path']): list(it['expert_actions']) for it in train_tasks
    }

    env_kwargs = build_placement_env_kwargs(locals())
    print(f'[env] CUDA PlacementEnv kwargs active: {env_kwargs}')

    viz_logger = _VisdomTrainLogger(
        enabled=bool(visdom_enabled),
        server=str(visdom_server),
        port=int(visdom_port),
        env=str(visdom_env),
        prefix=str(visdom_prefix),
    )
    visdom_interval = max(1, int(visdom_log_interval))

    runtime_cache = _BoardRuntimeCache(
        train_tasks,
        env_kwargs,
        sequence_policy=sequence_policy,
    )
    print('[cache] board runtime cache enabled: lazy task JSON + PlacementEnv + expert-prefix snapshots')

    model.train()

    stop_requested: Dict[str, Optional[int]] = {'signal': None}

    def _handle_stop(sig_num, _frame):
        stop_requested['signal'] = int(sig_num)
        print(f'[signal] received signal={int(sig_num)}; checkpoint will be saved at the next safe point.')

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    board_states: List[BoardCurriculumState] = [
        BoardCurriculumState(index=i, path=it['path'], kmax=len(it['expert_actions']))
        for i, it in enumerate(train_tasks)
    ]

    # Adaptive-k controller: global k, but readiness is tracked per board at the current k stage.
    k_curr = 1
    loss_ema: Optional[float] = None
    k_steps = 0
    probe_left = 0
    peak_ema: Optional[float] = None
    target_ema: Optional[float] = None

    schedule_total_steps = int(steps) if int(steps) > 0 else max(
        int(kmax_final) * max(1, int(adaptive_k_min_steps)),
        int(expert_mix_anneal_steps) if int(expert_mix_anneal_steps) > 0 else 0,
        int(warmup) + 1,
    )

    rr_ptr = 0
    main_step = 0
    step_cap_hit = False
    phase = 'main'
    replay_iter_done = 0
    replay_buffer: Optional[WeightedReplayBuffer] = None
    replay_best_score: Optional[float] = None
    replay_baselines_by_path: Dict[str, float] = {}

    if bool(resume):
        assert resume_ckpt is not None
        ckpt = resume_ckpt
        # A normal resume must be architecture-identical.  Partial loading is a
        # migration operation, not continuation, so fail immediately on drift.
        model.load_state_dict(ckpt['model_state'], strict=True)
        opt.load_state_dict(ckpt['optimizer_state'])
        if explicit_resume_lr is not None:
            _set_optimizer_learning_rate(opt, explicit_resume_lr)
            print(f'[resume] optimizer learning rate explicitly overridden to {float(explicit_resume_lr):.8g}')
        else:
            loaded_lrs = [float(group.get('lr', lr)) for group in opt.param_groups]
            print(f'[resume] optimizer learning rate restored from checkpoint: {loaded_lrs}')
        raw_saved_states = ckpt.get('board_states')
        if not isinstance(raw_saved_states, list):
            raise ValueError(
                "Resume checkpoint is missing a valid board_states list."
            )
        saved_states = [
            _board_state_from_dict(item)
            for item in raw_saved_states
        ]
        board_states = _restore_board_states_by_path(
            saved_states,
            board_states,
        )
        print(
            f"[resume] restored {len(board_states)} board curriculum states "
            "by normalized task path"
        )
        mt = ckpt.get('main_training', {})
        rr_ptr = int(mt.get('rr_ptr', 0))
        main_step = int(mt.get('main_steps_executed', 0))
        step_cap_hit = bool(mt.get('step_cap_hit', False))
        k_curr = int(mt.get('k_curr', 1))
        loss_ema = None if mt.get('loss_ema') is None else float(mt.get('loss_ema'))
        k_steps = int(mt.get('k_steps', 0))
        if bool(adaptive_k) and int(fixed_k_steps) > 0 and int(k_steps) <= 0:
            # Checkpoints created before fixed_k_steps did not advance k_steps.
            # Infer elapsed stage steps conservatively from per-board visits so resume can promote promptly.
            _growth_for_infer = [
                st for st in board_states
                if (not st.graduated) and int(st.kmax) > int(k_curr) and int(st.k_stage) == int(k_curr)
            ]
            if _growth_for_infer:
                k_steps = int(min(max(0, int(st.k_stage_visits)) for st in _growth_for_infer) * max(1, len(_growth_for_infer)))
                print(f'[resume] inferred fixed-k elapsed steps={k_steps} at k={int(k_curr)} from board visit counters')
        probe_left = int(mt.get('probe_left', 0))
        peak_ema = None if mt.get('peak_ema') is None else float(mt.get('peak_ema'))
        target_ema = None if mt.get('target_ema') is None else float(mt.get('target_ema'))
        phase = str(ckpt.get('phase', 'main'))
        replay_iter_done = int(ckpt.get('replay_iter_done', 0))
        replay_best_score = None if ckpt.get('replay_best_score') is None else float(ckpt.get('replay_best_score'))
        raw_replay_baselines = ckpt.get('replay_baselines_by_path', {})
        if raw_replay_baselines is None:
            raw_replay_baselines = {}
        if not isinstance(raw_replay_baselines, dict):
            raise ValueError(
                "Resume checkpoint replay_baselines_by_path must be a dict."
            )
        replay_baselines_by_path = _restore_replay_baselines_by_path(
            raw_replay_baselines,
            board_states,
        )
        replay_buffer = _restore_replay_buffer(ckpt.get('replay_buffer'))
        print(f'[resume] loaded checkpoint: {checkpoint_path}')
        print(f'[resume] phase={phase} main_step={main_step} replay_iter_done={replay_iter_done}')

    # Interval-based logging/checkpoint trackers.
    # Do not use `main_step % N == 0`, because main_step advances by the
    # actual active batch size. When only a few boards remain active, main_step
    # may never land exactly on a multiple of N, which makes logs/checkpoints
    # appear to stall even while training continues.
    last_log_step = int(main_step) - 200
    last_checkpoint_step = int(main_step)

    def _checkpoint_now(curr_phase: str, curr_step: int) -> None:
        if not checkpoint_path:
            return
        payload = _build_checkpoint_payload(
            model=model,
            opt=opt,
            obs_dim=obs_dim,
            max_tokens=max_tokens,
            sequence_policy=sequence_policy,
            board_states=board_states,
            kmax_final=kmax_final,
            warmup=warmup,
            steps=steps,
            max_main_steps=max_main_steps,
            adaptive_k=adaptive_k,
            fixed_k_steps=fixed_k_steps,
            curriculum_window_mode=curriculum_window_mode,
            adaptive_k_drop_ratio=adaptive_k_drop_ratio,
            adaptive_k_drop_ratio_first=adaptive_k_drop_ratio_first,
            adaptive_k_probe_steps=adaptive_k_probe_steps,
            adaptive_k_min_steps=adaptive_k_min_steps,
            adaptive_k_ema_beta=adaptive_k_ema_beta,
            adaptive_k_max_steps_per_k=adaptive_k_max_steps_per_k,
            adaptive_k_max_visits_target_slack=adaptive_k_max_visits_target_slack,
            adaptive_k_ready_ratio=adaptive_k_ready_ratio,
            adaptive_k_outlier_release_factor=adaptive_k_outlier_release_factor,
            board_full_ema_beta=board_full_ema_beta,
            board_full_probe_visits=board_full_probe_visits,
            board_full_drop_ratio=board_full_drop_ratio,
            board_full_min_visits=board_full_min_visits,
            board_full_plateau_patience=board_full_plateau_patience,
            board_full_plateau_rel_change=board_full_plateau_rel_change,
            board_full_max_visits=board_full_max_visits,
            min_spacing_mm=min_spacing_mm,
            env_alignment_bonus=env_alignment_bonus,
            env_edge_bonus=env_edge_bonus,
            env_edge_eps_mm=env_edge_eps_mm,
            reward_non_interface_edge_penalty=reward_non_interface_edge_penalty,
            reward_non_interface_edge_margin_mm=reward_non_interface_edge_margin_mm,
            reward_density_penalty=reward_density_penalty,
            reward_density_radius_mm=reward_density_radius_mm,
            reward_interior_penalty=reward_interior_penalty,
            reward_interior_margin_ratio=reward_interior_margin_ratio,
            objective_hpwl_weight=objective_hpwl_weight,
            objective_w_hpwl_weight=objective_w_hpwl_weight,
            objective_nslw_weight=objective_nslw_weight,
            objective_region_weight=objective_region_weight,
            objective_module_region_weight=objective_module_region_weight,
            objective_module_floorplan_weight=objective_module_floorplan_weight,
            module_region_bias=module_region_bias,
            module_region_margin_mm=module_region_margin_mm,
            module_floorplan_separation_mm=module_floorplan_separation_mm,
            module_floorplan_overlap_scale=module_floorplan_overlap_scale,
            module_floorplan_channel_scale=module_floorplan_channel_scale,
            module_floorplan_compact_scale=module_floorplan_compact_scale,
            module_floorplan_region_scale=module_floorplan_region_scale,
            objective_conn_weight=objective_conn_weight,
            objective_align_weight=objective_align_weight,
            objective_group_weight=objective_group_weight,
            objective_anchor_weight=objective_anchor_weight,
            objective_boundary_group_weight=objective_boundary_group_weight,
            objective_pitch_weight=objective_pitch_weight,
            objective_orientation_weight=objective_orientation_weight,
            objective_edge_clearance_weight=objective_edge_clearance_weight,
            objective_interior_weight=objective_interior_weight,
            objective_density_weight=objective_density_weight,
            objective_soft_spacing_weight=objective_soft_spacing_weight,
            objective_neatness_weight=objective_neatness_weight,
            edge_band_ratio=edge_band_ratio,
            edge_band_center_ratio=edge_band_center_ratio,
            soft_spacing_same_group_extra_mm=soft_spacing_same_group_extra_mm,
            soft_spacing_cross_group_extra_mm=soft_spacing_cross_group_extra_mm,
            soft_spacing_large_extra_mm=soft_spacing_large_extra_mm,
            same_group_density_scale=same_group_density_scale,
            critical_neighbor_density_scale=critical_neighbor_density_scale,
            anchor_group_density_scale=anchor_group_density_scale,
            large_pair_density_scale=large_pair_density_scale,
            teacher=teacher,
            region_cfg=region_cfg,
            geom_cfg=geom_cfg,
            expert_mix_start=expert_mix_start,
            expert_mix_end=expert_mix_end,
            expert_mix_anneal_steps=expert_mix_anneal_steps,
            replay_finetune=replay_finetune,
            replay_iters=replay_iters,
            replay_rollouts_per_iter=replay_rollouts_per_iter,
            replay_update_steps=replay_update_steps,
            replay_batch_size=replay_batch_size,
            replay_capacity=replay_capacity,
            replay_alpha=replay_alpha,
            replay_temp=replay_temp,
            replay_action_ce_coef=replay_action_ce_coef,
            replay_policy_gradient_coef=replay_policy_gradient_coef,
            replay_entropy_coef=replay_entropy_coef,
            replay_rollout_temperature=replay_rollout_temperature,
            replay_pg_policy_mode=replay_pg_policy_mode,
            replay_baseline_beta=replay_baseline_beta,
            replay_advantage_clip=replay_advantage_clip,
            replay_k=replay_k,
            main_step=main_step,
            rr_ptr=rr_ptr,
            step_cap_hit=step_cap_hit,
            k_curr=k_curr,
            loss_ema=loss_ema,
            k_steps=k_steps,
            probe_left=probe_left,
            peak_ema=peak_ema,
            target_ema=target_ema,
            phase=curr_phase,
            replay_iter_done=replay_iter_done,
            replay_buffer=replay_buffer,
            replay_best_score=replay_best_score,
            replay_baselines_by_path=replay_baselines_by_path,
        )
        latest_path, step_path, removed = _save_checkpoint(
            checkpoint_path=checkpoint_path,
            payload=payload,
            phase=curr_phase,
            step=curr_step,
            keep_last=keep_last_checkpoints,
        )
        msg = f'[checkpoint] saved latest={latest_path} step_copy={step_path}'
        if removed:
            msg += f' removed_old={len(removed)}'
        print(msg)

    if phase == 'main':
        while True:
            active_indices = [st.index for st in board_states if not st.graduated]
            if not active_indices:
                print(f'[main] all boards graduated at step={main_step}; enter replay/RL phase.')
                break

            if int(steps) > 0 and main_step >= int(steps):
                step_cap_hit = True
                print(
                    f'[main] reached steps={int(steps)} with {len(active_indices)} active boards left; '
                    'stop main training and continue to replay/RL.'
                )
                break

            if int(max_main_steps) > 0 and main_step >= int(max_main_steps):
                step_cap_hit = True
                print(
                    f'[main] reached max_main_steps={int(max_main_steps)} with {len(active_indices)} active boards left; '
                    'stop main training and continue to replay/RL.'
                )
                break

            # Batch processing: select multiple boards for true batched GPU computation.
            actual_bs = min(max(1, int(batch_size)), len(active_indices))
            start_idx = rr_ptr % len(active_indices)
            batch_indices = [active_indices[(start_idx + i) % len(active_indices)] for i in range(actual_bs)]

            rr_ptr += actual_bs
            main_step += actual_bs

            if adaptive_k:
                k = int(k_curr)
            else:
                k = schedule_kmax(main_step, schedule_total_steps, warmup, kmax_final)

            alpha_total = int(expert_mix_anneal_steps) if int(expert_mix_anneal_steps) > 0 else schedule_total_steps
            alpha_hold = max(int(warmup), int(0.25 * max(1, alpha_total)))
            alpha = schedule_mix(main_step, alpha_total, alpha_hold, expert_mix_start, expert_mix_end)

            # Process the boards with a single batched GPU forward per suffix step.
            batch_loss = []
            batch_envs: List[PlacementEnv] = []
            batch_expert_actions: List[List[Tuple[int, int, int]]] = []
            batch_task_paths: List[str] = []
            batch_k_effs: List[int] = []
            batch_window_starts: List[int] = []
            batch_window_steps: List[int] = []
            batch_board_states: List[BoardCurriculumState] = []

            for batch_idx in batch_indices:
                task_item = train_tasks[batch_idx]
                board_state = board_states[batch_idx]
                k_eff = min(int(k), int(board_state.kmax))
                T = int(board_state.kmax)
                steps_this = max(1, min(T, int(k_eff)))

                mode = str(curriculum_window_mode)
                if mode == 'mixed':
                    r = random.random()
                    if r < 0.50:
                        mode_eff = 'prefix'
                    elif r < 0.75:
                        mode_eff = 'suffix'
                    else:
                        mode_eff = 'random'
                else:
                    mode_eff = mode

                if mode_eff == 'prefix':
                    start_t = 0
                elif mode_eff == 'suffix':
                    start_t = max(0, T - steps_this)
                elif mode_eff == 'random':
                    start_t = random.randint(0, max(0, T - steps_this))
                else:
                    start_t = max(0, T - steps_this)

                env = runtime_cache.make_env_at_prefix(batch_idx, start_t)
                batch_envs.append(env)
                batch_expert_actions.append(task_item['expert_actions'])
                batch_task_paths.append(task_item['path'])
                batch_k_effs.append(k_eff)
                batch_window_starts.append(start_t)
                batch_window_steps.append(steps_this)
                batch_board_states.append(board_state)

            opt.zero_grad(set_to_none=True)
            _batch_log_loss, board_outcomes = rollout_suffix_loss_batched_boards(
                model=model,
                envs=batch_envs,
                expert_actions_list=batch_expert_actions,
                task_paths=batch_task_paths,
                k_list=batch_k_effs,
                alpha_expert=alpha,
                teacher=teacher,
                region_cfg=region_cfg,
                geom_cfg=geom_cfg,
                device=device_t,
                max_tokens=max_tokens,
                do_backward=True,
                grad_weight=1.0,
                prefix_replayed=True,
                window_starts=batch_window_starts,
                window_steps=batch_window_steps,
            )

            has_trainable_loss = _step_optimizer_if_trainable_batch(
                model,
                opt,
                board_outcomes,
                max_grad_norm=1.0,
            )
            if not has_trainable_loss:
                print(
                    f"[train-skip-batch] step={main_step} "
                    "no board produced a trainable loss; optimizer step skipped"
                )

            batch_suffix_complete_rate = float(
                sum(1 for outcome in board_outcomes if bool(outcome.suffix_complete))
                / max(1, len(board_outcomes))
            )
            batch_illegal_rate = float(
                sum(
                    1 for outcome in board_outcomes
                    if str(outcome.failure_reason or "").startswith("illegal_step")
                    or str(outcome.failure_reason or "").startswith("illegal_action")
                ) / max(1, len(board_outcomes))
            )
            batch_no_legal_rate = float(
                sum(
                    1 for outcome in board_outcomes
                    if "no_legal" in str(outcome.failure_reason or "")
                ) / max(1, len(board_outcomes))
            )

            valid_curriculum_visits = 0
            for outcome, board_state, k_eff in zip(
                board_outcomes,
                batch_board_states,
                batch_k_effs,
            ):
                if not outcome.curriculum_valid:
                    print(
                        f"[train-skip-board] board={board_state.path} "
                        f"k_eff={int(k_eff)} valid_steps={int(outcome.valid_steps)} "
                        f"terminated={bool(outcome.terminated)} "
                        f"reason={outcome.failure_reason or 'invalid_training_outcome'}"
                    )
                    continue

                assert outcome.loss is not None
                loss_val = float(outcome.loss)
                batch_loss.append(
                    (
                        torch.tensor(loss_val, device=device_t),
                        board_state,
                        k_eff,
                    )
                )
                valid_curriculum_visits += 1

            if adaptive_k:
                # Invalid/failed visits must not advance fixed-k curriculum.
                k_steps += int(valid_curriculum_visits)

            # Update states only for boards with a complete, valid suffix loss.
            for loss, board_state, k_eff in batch_loss:
                loss_val = float(loss.item())
                is_full_board = int(k_eff) >= int(board_state.kmax)

                if is_full_board:
                    graduated, grad_reason = _update_board_full_state(
                        board_state,
                        loss_val=loss_val,
                        main_step=main_step,
                        ema_beta=float(board_full_ema_beta),
                        probe_visits=int(board_full_probe_visits),
                        drop_ratio=float(board_full_drop_ratio),
                        min_visits=int(board_full_min_visits),
                        plateau_patience=int(board_full_plateau_patience),
                        plateau_rel_change=float(board_full_plateau_rel_change),
                        max_visits=int(board_full_max_visits),
                    )
                    if graduated:
                        fp = float(board_state.full_peak_ema) if board_state.full_peak_ema is not None else float('nan')
                        ft = float(board_state.full_target_ema) if board_state.full_target_ema is not None else float('nan')
                        fe = float(board_state.full_loss_ema) if board_state.full_loss_ema is not None else float('nan')
                        print(
                            f'[graduate] board={board_state.path} kmax={board_state.kmax} '
                            f'full_visits={board_state.full_visits} full_ema={fe:.4f} peak={fp:.4f} target={ft:.4f} '
                            f'reason={grad_reason} step={main_step}'
                        )

                if adaptive_k and int(fixed_k_steps) <= 0 and (not is_full_board) and (int(board_state.kmax) > int(k_curr)):
                    drop = float(adaptive_k_drop_ratio_first) if int(k_curr) <= 1 else float(adaptive_k_drop_ratio)
                    ready, ready_reason = _update_board_k_stage_state(
                        board_state,
                        k_stage=int(k_curr),
                        loss_val=loss_val,
                        ema_beta=float(adaptive_k_ema_beta),
                        probe_steps=int(adaptive_k_probe_steps),
                        drop_ratio=drop,
                        min_visits=int(adaptive_k_min_steps),
                        max_visits=int(adaptive_k_max_steps_per_k),
                        max_visits_target_slack=float(adaptive_k_max_visits_target_slack),
                        outlier_release_factor=float(adaptive_k_outlier_release_factor),
                    )
                    if ready:
                        ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                        kp = float(board_state.k_stage_peak_ema) if board_state.k_stage_peak_ema is not None else float('nan')
                        kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                        print(
                            f'[k-ready] board={board_state.path} k={int(board_state.k_stage)} '
                            f'visits={board_state.k_stage_visits} ema={ke:.4f} peak={kp:.4f} target={kt:.4f} '
                            f'reason={ready_reason} step={main_step}'
                        )
                    elif ready_reason == 'max_visits_hold':
                        ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                        kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                        print(
                            f'[k-hold] board={board_state.path} k={int(board_state.k_stage)} '
                            f'visits={board_state.k_stage_visits} ema={ke:.4f} target={kt:.4f} '
                            f'reason=max_visits_hold step={main_step}'
                        )
                    elif ready_reason == 'outlier_deferred':
                        ke = float(board_state.k_stage_loss_ema) if board_state.k_stage_loss_ema is not None else float('nan')
                        kt = float(board_state.k_stage_target_ema) if board_state.k_stage_target_ema is not None else float('nan')
                        print(
                            f'[k-defer] board={board_state.path} k={int(board_state.k_stage)} '
                            f'visits={board_state.k_stage_visits} ema={ke:.4f} target={kt:.4f} '
                            f'reason=outlier_release step={main_step}'
                        )

            # Recalculate growth_states after batch processing
            growth_states = [
                st for st in board_states
                if (not st.graduated) and (int(st.kmax) > int(k_curr))
            ]

            can_promote = False
            effective_ready_cnt = 0
            deferred_cnt = 0
            ready_ratio = 0.0
            fixed_k_promote = False
            if adaptive_k and growth_states and int(k_curr) < int(kmax_final):
                warmup_done = int(main_step) >= max(0, int(warmup))
                if int(fixed_k_steps) > 0:
                    fixed_k_promote = bool(warmup_done and int(k_steps) >= int(fixed_k_steps))
                    effective_ready_cnt = int(k_steps)
                    ready_ratio = float(k_steps) / float(max(1, int(fixed_k_steps)))
                    can_promote = fixed_k_promote
                else:
                    ready_cnt = sum(
                        1 for st in growth_states
                        if int(st.k_stage) == int(k_curr) and bool(st.k_stage_ready)
                    )
                    deferred_cnt = sum(
                        1 for st in growth_states
                        if int(st.k_stage) == int(k_curr) and bool(st.k_stage_deferred)
                    )
                    effective_ready_cnt = int(ready_cnt + deferred_cnt)
                    ready_ratio = float(effective_ready_cnt) / float(max(1, len(growth_states)))
                    can_promote = warmup_done and ready_ratio >= float(_clip01(adaptive_k_ready_ratio))

            if can_promote:
                old_k = int(k_curr)
                k_curr = min(int(kmax_final), old_k + 1)
                k_steps = 0
                probe_left = 0
                peak_ema = None
                target_ema = None
                for st in board_states:
                    if (not st.graduated) and (int(st.kmax) > old_k):
                        _reset_board_k_stage(st, int(k_curr))
                if int(fixed_k_steps) > 0:
                    active_left_promote = sum(0 if st.graduated else 1 for st in board_states)
                    grad_cnt_promote = len(board_states) - active_left_promote
                    last_loss_promote = float(batch_loss[-1][0].item()) if batch_loss else float('nan')
                    print(
                        f'[k-fixed] promote k {old_k}->{int(k_curr)} at step={main_step} '
                        f'(k_steps={int(effective_ready_cnt)}/{int(fixed_k_steps)} warmup_done={warmup_done}) '
                        f'active={active_left_promote}/{len(board_states)} graduated={grad_cnt_promote} '
                        f'last_loss={last_loss_promote:.4f}'
                    )
                else:
                    print(
                        f'[k-adapt] promote k {old_k}->{int(k_curr)} at step={main_step} '
                        f'(effective_ready={effective_ready_cnt}/{len(growth_states)} deferred={deferred_cnt} '
                        f'ready_ratio={ready_ratio:.3f} threshold={float(_clip01(adaptive_k_ready_ratio)):.3f} warmup_done={warmup_done})'
                    )


            if int(main_step) - int(last_log_step) >= 200:
                last_log_step = int(main_step)
                active_left = sum(0 if st.graduated else 1 for st in board_states)
                grad_cnt = len(board_states) - active_left
                if batch_loss:
                    last_board_state = batch_loss[-1][1]
                    last_k_eff = batch_loss[-1][2]
                    last_loss_val = float(batch_loss[-1][0].item())
                else:
                    last_board_state = None
                    last_k_eff = 0
                    last_loss_val = float('nan')
                if adaptive_k:
                    growth_states = [
                        st for st in board_states
                        if (not st.graduated) and (int(st.kmax) > int(k_curr))
                    ]
                    if int(fixed_k_steps) > 0:
                        print(
                            f'step={main_step} active={active_left}/{len(board_states)} graduated={grad_cnt} '
                            f'board={last_board_state.path if last_board_state else "N/A"} k={k} k_eff={last_k_eff} alpha_expert={alpha:.3f} '
                            f'loss={last_loss_val:.4f} fixed_k_steps={int(k_steps)}/{int(fixed_k_steps)} '
                            f'complete_rate={batch_suffix_complete_rate:.3f} illegal_rate={batch_illegal_rate:.3f} no_legal_rate={batch_no_legal_rate:.3f}'
                        )
                    else:
                        ready_cnt = sum(1 for st in growth_states if int(st.k_stage) == int(k_curr) and bool(st.k_stage_ready))
                        deferred_cnt = sum(1 for st in growth_states if int(st.k_stage) == int(k_curr) and bool(st.k_stage_deferred))
                        effective_ready_cnt = int(ready_cnt + deferred_cnt)
                        ke = float(last_board_state.k_stage_loss_ema) if last_board_state and last_board_state.k_stage_loss_ema is not None else float('nan')
                        kp = float(last_board_state.k_stage_peak_ema) if last_board_state and last_board_state.k_stage_peak_ema is not None else float('nan')
                        kt = float(last_board_state.k_stage_target_ema) if last_board_state and last_board_state.k_stage_target_ema is not None else float('nan')
                        print(
                            f'step={main_step} active={active_left}/{len(board_states)} graduated={grad_cnt} '
                            f'board={last_board_state.path if last_board_state else "N/A"} k={k} k_eff={last_k_eff} alpha_expert={alpha:.3f} '
                            f'loss={last_loss_val:.4f} ready={effective_ready_cnt}/{len(growth_states)} (strict={ready_cnt} deferred={deferred_cnt}) '
                            f'board_k_visits={last_board_state.k_stage_visits if last_board_state else 0} board_k_ema={ke:.4f} peak={kp:.4f} target={kt:.4f} '
                            f'complete_rate={batch_suffix_complete_rate:.3f} illegal_rate={batch_illegal_rate:.3f} no_legal_rate={batch_no_legal_rate:.3f}'
                        )
                else:
                    print(
                        f'step={main_step} active={active_left}/{len(board_states)} graduated={grad_cnt} '
                        f'board={last_board_state.path if last_board_state else "N/A"} k={k} k_eff={last_k_eff} alpha_expert={alpha:.3f} loss={last_loss_val:.4f} '
                        f'complete_rate={batch_suffix_complete_rate:.3f} illegal_rate={batch_illegal_rate:.3f} no_legal_rate={batch_no_legal_rate:.3f}'
                    )

                if last_board_state and last_k_eff >= int(last_board_state.kmax):
                    fp = float(last_board_state.full_peak_ema) if last_board_state.full_peak_ema is not None else float('nan')
                    ft = float(last_board_state.full_target_ema) if last_board_state.full_target_ema is not None else float('nan')
                    fe = float(last_board_state.full_loss_ema) if last_board_state.full_loss_ema is not None else float('nan')
                    print(
                        f'          full-board visits={last_board_state.full_visits} full_ema={fe:.4f} '
                        f'peak={fp:.4f} target={ft:.4f}'
                    )

                if viz_logger.enabled and (int(main_step) % int(visdom_interval) == 0):
                    last_full_ema = (
                        float(last_board_state.full_loss_ema)
                        if last_board_state and last_board_state.full_loss_ema is not None
                        else float('nan')
                    )
                    last_k_ema = (
                        float(last_board_state.k_stage_loss_ema)
                        if last_board_state and last_board_state.k_stage_loss_ema is not None
                        else float('nan')
                    )
                    viz_logger.scalars(
                        int(main_step),
                        {
                            'loss/main': last_loss_val,
                            'curriculum/k': int(k),
                            'curriculum/k_eff_last': int(last_k_eff),
                            'curriculum/active_boards': int(active_left),
                            'curriculum/graduated_boards': int(grad_cnt),
                            'curriculum/alpha_expert': float(alpha),
                            'curriculum/k_stage_ema_last': last_k_ema,
                            'curriculum/full_ema_last': last_full_ema,
                            'optim/lr': float(opt.param_groups[0].get('lr', lr)),
                            'train/suffix_complete_rate': float(batch_suffix_complete_rate),
                            'train/illegal_rate': float(batch_illegal_rate),
                            'train/no_legal_rate': float(batch_no_legal_rate),
                        },
                    )

            if (
                checkpoint_path
                and int(checkpoint_every_steps) > 0
                and int(main_step) - int(last_checkpoint_step) >= int(checkpoint_every_steps)
            ):
                last_checkpoint_step = int(main_step)
                _checkpoint_now('main', main_step)

            if stop_requested['signal'] is not None:
                print(f"[signal] graceful stop after main step={main_step}")
                _checkpoint_now('main', main_step)
                return

    # On-policy REINFORCE followed by weighted replay self-imitation
    phase = 'replay'
    if replay_finetune:
        print('=== replay RL: on-policy REINFORCE + weighted self-imitation ===')
        rb = replay_buffer if replay_buffer is not None else WeightedReplayBuffer(capacity=replay_capacity, alpha=replay_alpha)
        replay_k_eff = int(replay_k) if replay_k is not None else int(Tmax)

        # Build one board-specific reference objective from each complete expert
        # layout. Replay scores are fractional improvements over this reference,
        # so boards with different component counts/objective scales are comparable.
        replay_reference_obj_by_path: Dict[str, float] = {}
        expert_episode_by_path: Dict[str, Dict[str, Any]] = {}
        for board_idx, it in enumerate(train_tasks):
            cached = runtime_cache.get_item(board_idx)
            env = cached.base_env
            w2, h2 = env.grid_shape()
            flat = [
                _flatten_action(action[0], action[1], action[2], w2, h2)
                for action in it['expert_actions']
            ]
            full_prefix_idx = min(
                len(cached.prefix_states) - 1,
                len(it['expert_actions']),
            )
            env2 = _clone_env_from_dynamic_state(
                cached.base_env,
                cached.prefix_states[full_prefix_idx],
            )
            reference_obj = float(_final_objective(env2))
            task_path = str(it['path'])
            replay_reference_obj_by_path[task_path] = reference_obj
            expert_episode_by_path[task_path] = _normalize_replay_episode_score(
                {
                    'task_path': task_path,
                    'actions_flat': flat,
                    'total_return': -reference_obj,
                    'terminated': False,
                    'steps': int(len(flat)),
                },
                reference_obj,
            )

        if replay_buffer is None or len(rb) == 0:
            for task_path, expert_episode in expert_episode_by_path.items():
                rb.add(
                    expert_episode,
                    priority=_replay_priority_from_score(
                        float(expert_episode['score']),
                        replay_temp,
                    ),
                )
        else:
            # Migrate old checkpoint items that used score=-absolute_objective,
            # and recompute every saved priority from the normalized score.
            migrated_items: List[Dict[str, Any]] = []
            migrated_priorities: List[float] = []
            for item in rb.items:
                migrated = _normalize_replay_episode_for_path(
                    item,
                    replay_reference_obj_by_path,
                    context="replay checkpoint migration",
                )
                migrated_items.append(migrated)
                migrated_priorities.append(
                    _replay_priority_from_score(
                        float(migrated['score']),
                        replay_temp,
                    )
                )
            rb.items = migrated_items
            rb.priorities = migrated_priorities
            print(
                f'[replay] migrated/normalized {len(rb.items)} buffered episodes '
                f'to {REPLAY_SCORE_KIND}'
            )

        _require_normalized_replay_episodes(
            rb.items,
            context="replay buffer best-score scan",
        )
        best_score = (
            max(float(item["score"]) for item in rb.items)
            if len(rb)
            else -1e9
        )

        for it_i in range(replay_iter_done, replay_iters):
            # Keep the on-policy distribution reproducible between trajectory
            # collection and REINFORCE reconstruction. Exploration comes from
            # action sampling, not an unrecorded dropout mask. Always restore
            # train mode before replay self-imitation, including disabled-RL or
            # exception paths.
            fresh_rollout_episodes: List[Dict[str, Any]] = []
            fresh_policy_episodes: List[Dict[str, Any]] = []
            replay_rollout_complete_rate: Optional[float] = None
            replay_rollout_illegal_rate: Optional[float] = None
            replay_rollout_no_legal_rate: Optional[float] = None
            replay_rollout_raw_objective_mean: Optional[float] = None
            policy_gradient_loss_value: Optional[float] = None
            policy_advantage_mean: Optional[float] = None
            policy_advantage_std: Optional[float] = None
            try:
                model.eval()
                if int(replay_rollouts_per_iter) > 0:
                    selected_rollouts = [
                        random.choice(train_tasks)
                        for _ in range(int(replay_rollouts_per_iter))
                    ]
                    replay_rollout_bs = max(
                        1,
                        min(int(batch_size), len(selected_rollouts)),
                    )
                    for s0 in range(0, len(selected_rollouts), replay_rollout_bs):
                        eps = rollout_episodes_batched(
                            model,
                            selected_rollouts[s0:s0 + replay_rollout_bs],
                            device_t,
                            teacher,
                            region_cfg,
                            env_kwargs=env_kwargs,
                            runtime_cache=runtime_cache,
                            max_tokens=max_tokens,
                            sample_actions=True,
                            sampling_temperature=replay_rollout_temperature,
                        )
                        for ep in eps:
                            normalized_ep = _normalize_replay_episode_for_path(
                                ep,
                                replay_reference_obj_by_path,
                                context="fresh replay rollout",
                            )
                            fresh_rollout_episodes.append(normalized_ep)
                            if ep['steps'] > 0:
                                fresh_policy_episodes.append(normalized_ep)
                                # Self-imitation buffer keeps complete trajectories only;
                                # failed/partial rollouts are still used immediately by
                                # REINFORCE as negative-advantage samples.
                                if bool(normalized_ep.get('complete', False)):
                                    rb.add(
                                        normalized_ep,
                                        priority=_replay_priority_from_score(
                                            float(normalized_ep['score']),
                                            replay_temp,
                                        ),
                                    )
                                    best_score = max(
                                        best_score,
                                        float(normalized_ep['score']),
                                    )

                rollout_stats = _summarize_replay_rollout_episodes(
                    fresh_rollout_episodes
                )
                replay_rollout_complete_rate = rollout_stats["complete_rate"]
                replay_rollout_illegal_rate = rollout_stats["illegal_rate"]
                replay_rollout_no_legal_rate = rollout_stats["no_legal_rate"]
                replay_rollout_raw_objective_mean = rollout_stats["raw_objective_mean"]

                if (
                    fresh_policy_episodes
                    and (
                        float(replay_policy_gradient_coef) != 0.0
                        or float(replay_entropy_coef) != 0.0
                    )
                ):
                    if not all(
                        bool(ep.get('sampled_policy', False))
                        for ep in fresh_policy_episodes
                    ):
                        raise ValueError(
                            "REINFORCE update requires freshly sampled on-policy episodes."
                        )

                    policy_advantages = (
                        _compute_replay_advantages_and_update_baselines(
                            fresh_policy_episodes,
                            replay_baselines_by_path,
                            baseline_beta=replay_baseline_beta,
                            advantage_clip=replay_advantage_clip,
                        )
                    )
                    advantage_array = np.asarray(
                        policy_advantages,
                        dtype=np.float64,
                    )
                    policy_advantage_mean = float(advantage_array.mean())
                    policy_advantage_std = float(
                        advantage_array.std(ddof=0)
                    )

                    opt.zero_grad(set_to_none=True)
                    policy_gradient_loss_value = 0.0
                    policy_batch_size = max(
                        1,
                        min(
                            int(replay_batch_size),
                            len(fresh_policy_episodes),
                        ),
                    )
                    total_fresh = float(len(fresh_policy_episodes))
                    for pg_start in range(
                        0,
                        len(fresh_policy_episodes),
                        policy_batch_size,
                    ):
                        pg_end = pg_start + policy_batch_size
                        pg_batch = fresh_policy_episodes[pg_start:pg_end]
                        pg_advantages = policy_advantages[pg_start:pg_end]
                        pg_weights = [
                            1.0 / total_fresh
                            for _ in pg_batch
                        ]
                        pg_loss = replay_update_loss_on_episodes_batched(
                            model,
                            pg_batch,
                            k=replay_k_eff,
                            teacher=teacher,
                            region_cfg=region_cfg,
                            geom_cfg=geom_cfg,
                            expert_actions_by_path=expert_actions_by_path,
                            device=device_t,
                            action_ce_coef=0.0,
                            env_kwargs=env_kwargs,
                            runtime_cache=runtime_cache,
                            sample_weights=pg_weights,
                            advantages=pg_advantages,
                            distill_coef=0.0,
                            policy_gradient_coef=float(
                                replay_policy_gradient_coef
                            ),
                            entropy_coef=float(replay_entropy_coef),
                            rollout_temperature=float(
                                replay_rollout_temperature
                            ),
                            pg_policy_mode=str(replay_pg_policy_mode),
                            do_backward=True,
                            max_tokens=max_tokens,
                        )
                        policy_gradient_loss_value += float(pg_loss.item())

                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            finally:
                # Distillation/self-imitation remains a standard training-mode
                # update after the fresh policy phase. This also covers the
                # case where RL coefficients are zero but rollout collection
                # is still enabled.
                model.train()
            upd_losses: List[float] = []
            for _ in range(replay_update_steps):
                batch = rb.sample(replay_batch_size)
                if not batch:
                    continue
                batch = _require_normalized_replay_episodes(
                    batch,
                    context="replay self-imitation sample",
                )

                scores = np.array(
                    [float(ep["score"]) for ep in batch],
                    dtype=np.float64,
                )
                wts = _stable_replay_batch_weights(
                    scores,
                    replay_temp,
                )

                opt.zero_grad(set_to_none=True)

                sample_weights = [float(wi) / max(1, len(batch)) for wi in wts.tolist()]
                loss_ep = replay_update_loss_on_episodes_batched(
                    model,
                    batch,
                    k=replay_k_eff,
                    teacher=teacher,
                    region_cfg=region_cfg,
                    geom_cfg=geom_cfg,
                    expert_actions_by_path=expert_actions_by_path,
                    device=device_t,
                    action_ce_coef=float(replay_action_ce_coef),
                    env_kwargs=env_kwargs,
                    runtime_cache=runtime_cache,
                    sample_weights=sample_weights,
                    do_backward=True,
                    max_tokens=max_tokens,
                )
                loss_b_val = float(loss_ep.item())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                upd_losses.append(float(loss_b_val))

            replay_iter_done = int(it_i + 1)
            replay_buffer = rb
            replay_best_score = float(best_score)

            if upd_losses or policy_gradient_loss_value is not None:
                replay_loss_mean = (
                    float(np.mean(upd_losses))
                    if upd_losses
                    else float('nan')
                )
                pg_text = (
                    "none"
                    if policy_gradient_loss_value is None
                    else f"{policy_gradient_loss_value:.4f}"
                )
                print(
                    f'replay iter={it_i + 1}/{replay_iters} buffer={len(rb)} '
                    f'best_relative_improvement={best_score:.4f} '
                    f'pg_loss={pg_text} pg_mode={replay_pg_policy_mode} '
                    f'self_imitation_loss={replay_loss_mean:.4f} '
                    f'complete_rate={(replay_rollout_complete_rate if replay_rollout_complete_rate is not None else float("nan")):.3f} '
                    f'illegal_rate={(replay_rollout_illegal_rate if replay_rollout_illegal_rate is not None else float("nan")):.3f} '
                    f'no_legal_rate={(replay_rollout_no_legal_rate if replay_rollout_no_legal_rate is not None else float("nan")):.3f} '
                    f'raw_obj_mean={(replay_rollout_raw_objective_mean if replay_rollout_raw_objective_mean is not None else float("nan")):.3f}'
                )
                if viz_logger.enabled:
                    replay_scalars: Dict[str, Any] = {
                        'replay/update_loss': replay_loss_mean,
                        'replay/buffer_size': int(len(rb)),
                        'replay/best_relative_improvement': float(best_score),
                        'replay/iter': int(it_i + 1),
                        'replay/rollout_complete_rate': float(replay_rollout_complete_rate or 0.0),
                        'replay/rollout_illegal_rate': float(replay_rollout_illegal_rate or 0.0),
                        'replay/rollout_no_legal_rate': float(replay_rollout_no_legal_rate or 0.0),
                        'replay/rollout_raw_objective_mean': float(replay_rollout_raw_objective_mean or 0.0),
                    }
                    if policy_gradient_loss_value is not None:
                        replay_scalars.update(
                            {
                                'replay/policy_gradient_loss': float(
                                    policy_gradient_loss_value
                                ),
                                'replay/advantage_mean': float(
                                    policy_advantage_mean or 0.0
                                ),
                                'replay/advantage_std': float(
                                    policy_advantage_std or 0.0
                                ),
                            }
                        )
                    viz_logger.scalars(
                        int(main_step + it_i + 1),
                        replay_scalars,
                    )

            if checkpoint_path:
                _checkpoint_now('replay', main_step)

            if stop_requested['signal'] is not None:
                print(f"[signal] graceful stop during replay iter={it_i + 1}")
                _checkpoint_now('replay', main_step)
                return

    final_payload = _build_checkpoint_payload(
        model=model,
        opt=opt,
        obs_dim=obs_dim,
        max_tokens=max_tokens,
        sequence_policy=sequence_policy,
        board_states=board_states,
        kmax_final=kmax_final,
        warmup=warmup,
        steps=steps,
        max_main_steps=max_main_steps,
        adaptive_k=adaptive_k,
        fixed_k_steps=fixed_k_steps,
        curriculum_window_mode=curriculum_window_mode,
        adaptive_k_drop_ratio=adaptive_k_drop_ratio,
        adaptive_k_drop_ratio_first=adaptive_k_drop_ratio_first,
        adaptive_k_probe_steps=adaptive_k_probe_steps,
        adaptive_k_min_steps=adaptive_k_min_steps,
        adaptive_k_ema_beta=adaptive_k_ema_beta,
        adaptive_k_max_steps_per_k=adaptive_k_max_steps_per_k,
        adaptive_k_max_visits_target_slack=adaptive_k_max_visits_target_slack,
        adaptive_k_ready_ratio=adaptive_k_ready_ratio,
        adaptive_k_outlier_release_factor=adaptive_k_outlier_release_factor,
        board_full_ema_beta=board_full_ema_beta,
        board_full_probe_visits=board_full_probe_visits,
        board_full_drop_ratio=board_full_drop_ratio,
        board_full_min_visits=board_full_min_visits,
        board_full_plateau_patience=board_full_plateau_patience,
        board_full_plateau_rel_change=board_full_plateau_rel_change,
        board_full_max_visits=board_full_max_visits,
        min_spacing_mm=min_spacing_mm,
        env_alignment_bonus=env_alignment_bonus,
        env_edge_bonus=env_edge_bonus,
        env_edge_eps_mm=env_edge_eps_mm,
        reward_non_interface_edge_penalty=reward_non_interface_edge_penalty,
        reward_non_interface_edge_margin_mm=reward_non_interface_edge_margin_mm,
        reward_density_penalty=reward_density_penalty,
        reward_density_radius_mm=reward_density_radius_mm,
        reward_interior_penalty=reward_interior_penalty,
        reward_interior_margin_ratio=reward_interior_margin_ratio,
        objective_hpwl_weight=objective_hpwl_weight,
        objective_w_hpwl_weight=objective_w_hpwl_weight,
        objective_nslw_weight=objective_nslw_weight,
        objective_region_weight=objective_region_weight,
        objective_module_region_weight=objective_module_region_weight,
        objective_module_floorplan_weight=objective_module_floorplan_weight,
        module_region_bias=module_region_bias,
        module_region_margin_mm=module_region_margin_mm,
        module_floorplan_separation_mm=module_floorplan_separation_mm,
        module_floorplan_overlap_scale=module_floorplan_overlap_scale,
        module_floorplan_channel_scale=module_floorplan_channel_scale,
        module_floorplan_compact_scale=module_floorplan_compact_scale,
        module_floorplan_region_scale=module_floorplan_region_scale,
        objective_conn_weight=objective_conn_weight,
        objective_align_weight=objective_align_weight,
        objective_group_weight=objective_group_weight,
        objective_anchor_weight=objective_anchor_weight,
        objective_boundary_group_weight=objective_boundary_group_weight,
        objective_pitch_weight=objective_pitch_weight,
        objective_orientation_weight=objective_orientation_weight,
        objective_edge_clearance_weight=objective_edge_clearance_weight,
        objective_interior_weight=objective_interior_weight,
        objective_density_weight=objective_density_weight,
        objective_soft_spacing_weight=objective_soft_spacing_weight,
        objective_neatness_weight=objective_neatness_weight,
        edge_band_ratio=edge_band_ratio,
        edge_band_center_ratio=edge_band_center_ratio,
        soft_spacing_same_group_extra_mm=soft_spacing_same_group_extra_mm,
        soft_spacing_cross_group_extra_mm=soft_spacing_cross_group_extra_mm,
        soft_spacing_large_extra_mm=soft_spacing_large_extra_mm,
        same_group_density_scale=same_group_density_scale,
        critical_neighbor_density_scale=critical_neighbor_density_scale,
        anchor_group_density_scale=anchor_group_density_scale,
        large_pair_density_scale=large_pair_density_scale,
        teacher=teacher,
        region_cfg=region_cfg,
        geom_cfg=geom_cfg,
        expert_mix_start=expert_mix_start,
        expert_mix_end=expert_mix_end,
        expert_mix_anneal_steps=expert_mix_anneal_steps,
        replay_finetune=replay_finetune,
        replay_iters=replay_iters,
        replay_rollouts_per_iter=replay_rollouts_per_iter,
        replay_update_steps=replay_update_steps,
        replay_batch_size=replay_batch_size,
        replay_capacity=replay_capacity,
        replay_alpha=replay_alpha,
        replay_temp=replay_temp,
        replay_action_ce_coef=replay_action_ce_coef,
        replay_policy_gradient_coef=replay_policy_gradient_coef,
        replay_entropy_coef=replay_entropy_coef,
        replay_rollout_temperature=replay_rollout_temperature,
        replay_pg_policy_mode=replay_pg_policy_mode,
        replay_baseline_beta=replay_baseline_beta,
        replay_advantage_clip=replay_advantage_clip,
        replay_k=replay_k,
        main_step=main_step,
        rr_ptr=rr_ptr,
        step_cap_hit=step_cap_hit,
        k_curr=k_curr,
        loss_ema=loss_ema,
        k_steps=k_steps,
        probe_left=probe_left,
        peak_ema=peak_ema,
        target_ema=target_ema,
        phase='done',
        replay_iter_done=replay_iter_done,
        replay_buffer=replay_buffer,
        replay_best_score=replay_best_score,
        replay_baselines_by_path=replay_baselines_by_path,
    )
    _atomic_torch_save(final_payload, save_path)
    if checkpoint_path:
        _checkpoint_now('done', main_step)
    print(f'saved: {save_path}')
