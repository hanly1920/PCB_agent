from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Literal
from pydantic import Field
from .schemas.base import AgentModel

class LLMConfig(AgentModel):
    provider: Literal["ollama", "openai", "mock", "disabled"] = "mock"
    endpoint: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    api_key: str | None = None
    timeout_sec: int = 120
    temperature: float = 0.1

class PlacementConfig(AgentModel):
    checkpoint_path: Path | None = None
    device: str = "cuda"
    mock: bool = False
    strict_state_dict: bool = True
    beam_width: int = 1
    beam_topk: int = 16
    layout_preset: str = "checkpoint"

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

class AgentConfig(AgentModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    kicad: KiCadConfig = Field(default_factory=KiCadConfig)
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
            import yaml
            data = yaml.safe_load(text) or {}
        return cls.model_validate(data)
