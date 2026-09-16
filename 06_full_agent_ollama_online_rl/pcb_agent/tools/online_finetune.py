from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..config import OnlineFinetuneConfig, PlacementConfig
from ..schemas import LayoutDSL, ToolResult


def _candidate_pose(value: dict[str, Any]) -> tuple[list[float], float] | None:
    try:
        x = float(value.get("x", value.get("cx")))
        y = float(value.get("y", value.get("cy")))
    except Exception:
        return None
    rot = float(value.get("rotation", value.get("rot", value.get("rot_deg", 0.0)))) % 360.0
    return [x, y], rot


def _optimizer_state_to_cpu(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _optimizer_state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_optimizer_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_state_to_cpu(v) for v in value)
    return value


class OnlinePolicyFinetuner:
    """Single-board online replay/RL update used by the agent loop.

    The tool does not overwrite the base checkpoint.  It writes a run-local
    session checkpoint and returns that path so the next placement round can
    use the updated policy.
    """

    name = "online_policy_finetuner"

    def __init__(self, config: OnlineFinetuneConfig, placement: PlacementConfig):
        self.config = config
        self.placement = placement

    def run(
        self,
        *,
        task_json_path: str | Path,
        selected_candidate: dict[str, Any],
        dsl: LayoutDSL,
        output_dir: str | Path,
        round_index: int,
        policy_updates: int | None = None,
    ) -> ToolResult:
        if not self.config.enabled:
            return ToolResult.success(self.name, "Online policy finetune disabled", metrics={"policy_updates": 0})
        if self.placement.mock:
            return ToolResult.success(self.name, "Online policy finetune skipped in mock placement mode", metrics={"policy_updates": 0})
        ckpt_path = Path(self.placement.checkpoint_path or "")
        if not ckpt_path.exists():
            return ToolResult.failure(self.name, f"Checkpoint not found for online finetune: {ckpt_path}")

        requested_updates = int(
            policy_updates
            if policy_updates is not None
            else (dsl.iteration_policy.policy_updates or self.config.default_policy_updates)
        )
        if requested_updates <= 0:
            return ToolResult.success(self.name, "Online policy finetune skipped because policy_updates <= 0", metrics={"policy_updates": 0})

        try:
            import torch
            from torch import nn
            from torch.optim import AdamW

            from scripts.step3_train_masked import _expert_actions_from_json
            from pcbplace.infer import apply_layout_objective_preset, clear_inference_model_cache, load_model
            from pcbplace.train import (
                GeometryDistillConfig,
                _BoardRuntimeCache,
                _clone_env_from_dynamic_state,
                _compute_replay_advantages_and_update_baselines,
                _final_objective,
                _flatten_action,
                _normalize_replay_episode_for_path,
                _replay_priority_from_score,
                _stable_replay_batch_weights,
                _summarize_replay_rollout_episodes,
                build_placement_env_kwargs,
                replay_update_loss_on_episodes_batched,
                rollout_episodes_batched,
            )
        except Exception as exc:
            return ToolResult.failure(self.name, f"Online finetune dependencies unavailable: {exc}")

        device_t = torch.device(self.placement.device)
        if device_t.type != "cuda":
            return ToolResult.failure(self.name, f"Online policy finetune is CUDA-only; got device={self.placement.device!r}")
        if not torch.cuda.is_available():
            return ToolResult.failure(self.name, "Online policy finetune requires CUDA, but torch.cuda.is_available() is False")

        try:
            out_dir = Path(output_dir)
            ckpt_dir = out_dir / self.config.save_subdir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            expert_task = self._write_candidate_as_expert_task(
                Path(task_json_path),
                selected_candidate,
                ckpt_dir / f"round_{round_index:02d}_online_expert.json",
            )

            raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if not isinstance(raw_ckpt, dict) or "model_state" not in raw_ckpt:
                raise ValueError(f"Checkpoint must contain model_state: {ckpt_path}")

            model, metadata, teacher, region_cfg = load_model(
                str(ckpt_path),
                device=str(device_t),
                strict_state_dict=bool(self.placement.strict_state_dict),
            )
            model.train()

            env_config = dict(metadata.get("env_config") or raw_ckpt.get("env_config") or {})
            env_kwargs = build_placement_env_kwargs(env_config)
            env_kwargs = apply_layout_objective_preset(
                env_kwargs,
                self.placement.layout_preset,
                dsl.objective_overrides(),
            )
            sequence_policy = str(metadata.get("sequence_policy") or raw_ckpt.get("sequence_policy") or "rebuild")
            max_tokens = int(metadata.get("max_tokens") or raw_ckpt.get("max_tokens") or 128)
            replay_temperature = max(1.0e-6, float(dsl.iteration_policy.replay_temperature or 1.0))
            rollout_budget = max(
                1,
                int(dsl.iteration_policy.rollout_budget or self.config.replay_rollouts_per_iter),
            )

            expert_actions = _expert_actions_from_json(
                str(expert_task),
                snap_radius=int(self.config.expert_snap_radius),
                global_fallback=bool(self.config.expert_snap_global_fallback),
                env_kwargs=env_kwargs,
                device=device_t,
                sequence_policy=sequence_policy,
            )
            if not expert_actions:
                raise ValueError("Online finetune produced no expert actions from selected candidate")

            train_task = {"path": str(expert_task), "expert_actions": expert_actions}
            runtime_cache = _BoardRuntimeCache([train_task], env_kwargs, sequence_policy=sequence_policy)
            cached = runtime_cache.get_item(0)
            env = cached.base_env
            w, h = env.grid_shape()
            flat_expert = [_flatten_action(a[0], a[1], a[2], w, h) for a in expert_actions]
            full_prefix_idx = min(len(cached.prefix_states) - 1, len(expert_actions))
            expert_env = _clone_env_from_dynamic_state(cached.base_env, cached.prefix_states[full_prefix_idx])
            reference_obj = float(_final_objective(expert_env))
            reference_obj_by_path = {str(expert_task): reference_obj}
            expert_episode = _normalize_replay_episode_for_path(
                {
                    "task_path": str(expert_task),
                    "actions_flat": flat_expert,
                    "total_return": -reference_obj,
                    "final_obj": reference_obj,
                    "terminated": False,
                    "complete": True,
                    "steps": len(flat_expert),
                    "source": "agent_selected_candidate",
                },
                reference_obj_by_path,
                context="online expert bootstrap",
            )

            opt = AdamW(model.parameters(), lr=float(self.config.learning_rate))
            geom_cfg = GeometryDistillConfig()
            expert_actions_by_path = {str(expert_task): expert_actions}
            baselines_by_path: dict[str, float] = {}
            metrics: dict[str, Any] = {
                "policy_updates": int(requested_updates),
                "rollout_budget": int(rollout_budget),
                "reference_objective": float(reference_obj),
                "replay_temperature": float(replay_temperature),
                "pg_losses": [],
                "self_imitation_losses": [],
                "rollout_complete_rates": [],
                "rollout_illegal_rates": [],
                "rollout_no_legal_rates": [],
            }

            random.seed(1009 + int(round_index))
            np.random.seed(1009 + int(round_index))
            torch.manual_seed(1009 + int(round_index))
            torch.cuda.manual_seed_all(1009 + int(round_index))

            for update_i in range(int(requested_updates)):
                fresh = []
                model.eval()
                with torch.enable_grad():
                    raw_eps = rollout_episodes_batched(
                        model,
                        [train_task for _ in range(int(rollout_budget))],
                        device_t,
                        teacher,
                        region_cfg,
                        env_kwargs=env_kwargs,
                        runtime_cache=runtime_cache,
                        max_tokens=max_tokens,
                        sample_actions=True,
                        sampling_temperature=replay_temperature,
                    )
                for ep in raw_eps:
                    fresh.append(
                        _normalize_replay_episode_for_path(
                            ep,
                            reference_obj_by_path,
                            context="online fresh replay rollout",
                        )
                    )
                stats = _summarize_replay_rollout_episodes(fresh)
                metrics["rollout_complete_rates"].append(stats.get("complete_rate"))
                metrics["rollout_illegal_rates"].append(stats.get("illegal_rate"))
                metrics["rollout_no_legal_rates"].append(stats.get("no_legal_rate"))

                policy_eps = [ep for ep in fresh if int(ep.get("steps") or 0) > 0 and bool(ep.get("sampled_policy"))]
                if policy_eps and (float(self.config.replay_policy_gradient_coef) != 0.0 or float(self.config.replay_entropy_coef) != 0.0):
                    advantages = _compute_replay_advantages_and_update_baselines(
                        policy_eps,
                        baselines_by_path,
                        baseline_beta=float(self.config.replay_baseline_beta),
                        advantage_clip=float(self.config.replay_advantage_clip),
                    )
                    opt.zero_grad(set_to_none=True)
                    pg_loss = replay_update_loss_on_episodes_batched(
                        model,
                        policy_eps,
                        k=len(expert_actions),
                        teacher=teacher,
                        region_cfg=region_cfg,
                        geom_cfg=geom_cfg,
                        expert_actions_by_path=expert_actions_by_path,
                        device=device_t,
                        action_ce_coef=0.0,
                        env_kwargs=env_kwargs,
                        runtime_cache=runtime_cache,
                        sample_weights=[1.0 / float(max(1, len(policy_eps))) for _ in policy_eps],
                        advantages=advantages,
                        distill_coef=0.0,
                        policy_gradient_coef=float(self.config.replay_policy_gradient_coef),
                        entropy_coef=float(self.config.replay_entropy_coef),
                        rollout_temperature=replay_temperature,
                        pg_policy_mode=str(self.config.replay_pg_policy_mode),
                        do_backward=True,
                        max_tokens=max_tokens,
                    )
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    metrics["pg_losses"].append(float(pg_loss.detach().item()))

                model.train()
                complete_eps = [ep for ep in fresh if bool(ep.get("complete", False))]
                replay_pool = [expert_episode] + complete_eps
                replay_pool.sort(key=lambda ep: float(ep.get("score", -1.0e9)), reverse=True)
                batch_size = max(1, int(self.config.replay_batch_size))
                update_steps = max(1, int(self.config.replay_update_steps))
                for _step in range(update_steps):
                    batch = replay_pool[:batch_size]
                    scores = np.asarray([float(ep.get("score", 0.0)) for ep in batch], dtype=np.float64)
                    weights = _stable_replay_batch_weights(scores, replay_temperature)
                    opt.zero_grad(set_to_none=True)
                    si_loss = replay_update_loss_on_episodes_batched(
                        model,
                        batch,
                        k=len(expert_actions),
                        teacher=teacher,
                        region_cfg=region_cfg,
                        geom_cfg=geom_cfg,
                        expert_actions_by_path=expert_actions_by_path,
                        device=device_t,
                        action_ce_coef=float(self.config.replay_action_ce_coef),
                        env_kwargs=env_kwargs,
                        runtime_cache=runtime_cache,
                        sample_weights=[float(x) / float(max(1, len(batch))) for x in weights.tolist()],
                        distill_coef=1.0,
                        policy_gradient_coef=0.0,
                        entropy_coef=0.0,
                        rollout_temperature=replay_temperature,
                        pg_policy_mode=str(self.config.replay_pg_policy_mode),
                        do_backward=True,
                        max_tokens=max_tokens,
                    )
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    metrics["self_imitation_losses"].append(float(si_loss.detach().item()))

            new_ckpt = ckpt_dir / f"round_{round_index:02d}_online_policy.pt"
            payload = dict(raw_ckpt)
            payload["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload["optimizer_state"] = _optimizer_state_to_cpu(opt.state_dict())
            payload["phase"] = "agent_online_finetune"
            payload["online_finetune"] = {
                "base_checkpoint": str(ckpt_path),
                "round_index": int(round_index),
                "task_json_path": str(task_json_path),
                "expert_task_path": str(expert_task),
                "policy_updates": int(requested_updates),
                "learning_rate": float(self.config.learning_rate),
                "replay_update_steps": int(self.config.replay_update_steps),
                "replay_rollouts_per_iter": int(rollout_budget),
                "replay_temperature": float(replay_temperature),
                "policy_gradient_coef": float(self.config.replay_policy_gradient_coef),
                "action_ce_coef": float(self.config.replay_action_ce_coef),
                "entropy_coef": float(self.config.replay_entropy_coef),
            }
            self._atomic_torch_save(torch, payload, new_ckpt)
            clear_inference_model_cache()
            metrics["checkpoint_size_bytes"] = int(new_ckpt.stat().st_size)
            return ToolResult.success(
                self.name,
                "Online policy finetune completed and session checkpoint saved",
                artifacts={
                    "checkpoint_path": str(new_ckpt),
                    "base_checkpoint_path": str(ckpt_path),
                    "expert_task_path": str(expert_task),
                },
                metrics=metrics,
            )
        except Exception as exc:
            return ToolResult.failure(self.name, str(exc))

    @staticmethod
    def _write_candidate_as_expert_task(task_json_path: Path, selected_candidate: dict[str, Any], output_path: Path) -> Path:
        data = json.loads(Path(task_json_path).read_text(encoding="utf-8"))
        placements = selected_candidate.get("placements") or {}
        missing: list[str] = []
        for comp in data.get("components", []):
            ref = str(comp.get("ref") or "")
            if not ref:
                continue
            pose = _candidate_pose(placements.get(ref) or {})
            if pose is None:
                if not bool(comp.get("fixed")):
                    missing.append(ref)
                continue
            xy, rot = pose
            comp["expert"] = {"xy_mm": xy, "rot": float(rot)}
            comp["agent_online_expert"] = True
        if missing:
            raise ValueError("Selected candidate is missing placements for non-fixed refs: " + ", ".join(sorted(missing)))
        data.setdefault("meta", {})["agent_online_finetune"] = {
            "source": "selected_candidate",
            "candidate_id": selected_candidate.get("candidate_id"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    @staticmethod
    def _atomic_torch_save(torch_mod: Any, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            torch_mod.save(payload, str(tmp))
            os.replace(str(tmp), str(path))
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
