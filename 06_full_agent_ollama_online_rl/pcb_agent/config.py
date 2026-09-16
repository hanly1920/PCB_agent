from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Literal
import os
from pydantic import Field
from .schemas.base import AgentModel


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for the repository's simple config files.

    It supports top-level sections with two-space-indented scalar keys and
    inline comments. PyYAML is still used when installed.
    """
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            if value.strip():
                root[key] = _parse_scalar(value)
                current = None
            else:
                current = {}
                root[key] = current
            continue
        if current is None:
            continue
        key, _, value = line.strip().partition(":")
        current[key.strip()] = _parse_scalar(value)
    return root

class LLMConfig(AgentModel):
    provider: Literal["ollama", "openai", "mock", "disabled"] = "ollama"
    endpoint: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    api_key: str | None = None
    timeout_sec: int = 120
    temperature: float = 0.1

class PlacementConfig(AgentModel):
    checkpoint_path: Path | None = (
        Path(os.environ["PCB_AGENT_CHECKPOINT"])
        if os.environ.get("PCB_AGENT_CHECKPOINT")
        else None
    )
    device: str = "cuda"
    mock: bool = False
    strict_state_dict: bool = True
    beam_width: int = 1
    beam_topk: int = 16
    layout_preset: str = "checkpoint"

class OnlineFinetuneConfig(AgentModel):
    enabled: bool = True
    default_policy_updates: int = 1
    learning_rate: float = 1.0e-5
    replay_batch_size: int = 2
    replay_update_steps: int = 4
    replay_rollouts_per_iter: int = 8
    replay_policy_gradient_coef: float = 1.0
    replay_action_ce_coef: float = 0.1
    replay_entropy_coef: float = 0.01
    replay_baseline_beta: float = 0.90
    replay_advantage_clip: float = 1.0
    replay_pg_policy_mode: str = "shaped"
    expert_snap_radius: int = 6
    expert_snap_global_fallback: bool = True
    save_subdir: str = "online_checkpoints"

class RouterConfig(AgentModel):
    mode: Literal["freerouting", "none", "mock"] = "mock"
    java_bin: str = "java"
    freerouting_jar: Path | None = None
    passes: int = 50
    threads: int = 8
    timeout_sec: int = 1800
    export_dsn_cmd: list[str] = Field(default_factory=list)
    import_ses_cmd: list[str] = Field(default_factory=list)

class KiCadConfig(AgentModel):
    cli: str = "kicad-cli"
    drc_timeout_sec: int = 600
    mock_drc: bool = False

class WorkflowConfig(AgentModel):
    # A complete agent should use routing and DRC when the host provides them,
    # but it should still return a checked placement artifact when those
    # installation-specific tools are unavailable.
    run_routing: bool = True
    run_drc: bool = True
    auto_skip_unavailable_tools: bool = True
    accept_placement_without_drc: bool = True
    require_drc_clean: bool = False
    export_best_effort_board: bool = True
    llm_tuning_enabled: bool = True
    min_optimization_rounds: int = 2
    allow_llm_accept: bool = True

class AgentConfig(AgentModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    online_finetune: OnlineFinetuneConfig = Field(default_factory=OnlineFinetuneConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    kicad: KiCadConfig = Field(default_factory=KiCadConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    max_rounds: int = 3

    @classmethod
    def load(cls, path: str | Path | None) -> "AgentConfig":
        if path is None:
            return cls()
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data: dict[str, Any] = json.loads(text)
        else:
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ModuleNotFoundError:
                data = _simple_yaml_load(text)
        return cls.model_validate(data)
