from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import Field, field_validator
from .base import AgentModel

class DesignBundle(AgentModel):
    project_dir: Path
    board_path: Path
    task_json_path: Path | None = None
    netlist_path: Path | None = None
    footprint_map_path: Path | None = None
    rules_path: Path | None = None
    constraints_path: Path | None = None
    output_dir: Path | None = None
    user_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_dir", "board_path", "task_json_path", "netlist_path", "footprint_map_path", "rules_path", "constraints_path", "output_dir", mode="before")
    @classmethod
    def coerce_path(cls, value: Any) -> Any:
        return None if value in (None, "") else Path(value)

    def resolved(self, base: Path | None = None) -> "DesignBundle":
        base = (base or self.project_dir).expanduser().resolve()
        data = self.model_dump()
        for key in ("project_dir", "board_path", "task_json_path", "netlist_path", "footprint_map_path", "rules_path", "constraints_path", "output_dir"):
            value = data.get(key)
            if value is None:
                continue
            p = Path(value).expanduser()
            if not p.is_absolute():
                p = base / p
            data[key] = p.resolve()
        return DesignBundle.model_validate(data)

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "DesignBundle":
        bundle_dir = Path(bundle_dir).expanduser().resolve()
        manifest = None
        for name in ("bundle.yaml", "bundle.yml", "bundle.json"):
            candidate = bundle_dir / name
            if candidate.exists():
                manifest = candidate
                break
        if manifest is None:
            boards = sorted(bundle_dir.glob("*.kicad_pcb"))
            tasks = sorted(bundle_dir.glob("*.json"))
            if not boards:
                raise FileNotFoundError(f"No bundle manifest or .kicad_pcb found in {bundle_dir}")
            data = {"project_dir": str(bundle_dir), "board_path": boards[0].name}
            likely_tasks = [p for p in tasks if p.name != "bundle.json"]
            if likely_tasks:
                data["task_json_path"] = likely_tasks[0].name
        elif manifest.suffix == ".json":
            data = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise RuntimeError("PyYAML is required for YAML bundle manifests") from exc
            data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        data.setdefault("project_dir", str(bundle_dir))
        return cls.model_validate(data).resolved(bundle_dir)
