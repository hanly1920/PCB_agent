from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import Field, field_validator
from .base import AgentModel


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
        if bundle_dir.is_file():
            if bundle_dir.suffix.lower() == ".kicad_pcb":
                project_dir = bundle_dir.parent
                same_stem = bundle_dir.with_suffix(".json")
                task_path = same_stem if same_stem.exists() else None
                if task_path is None:
                    candidates = [
                        p for p in sorted(project_dir.glob("*.json"))
                        if p.name not in {"bundle.json", "state.json", "metrics.json"}
                    ]
                    task_path = candidates[0] if candidates else None
                data: dict[str, Any] = {
                    "project_dir": str(project_dir),
                    "board_path": str(bundle_dir),
                }
                if task_path is not None:
                    data["task_json_path"] = str(task_path)
                return cls.model_validate(data).resolved(project_dir)
            if bundle_dir.name in {"bundle.json", "bundle.yaml", "bundle.yml"}:
                project_dir = bundle_dir.parent
                if bundle_dir.suffix == ".json":
                    data = json.loads(bundle_dir.read_text(encoding="utf-8"))
                else:
                    try:
                        import yaml
                        data = yaml.safe_load(bundle_dir.read_text(encoding="utf-8")) or {}
                    except ModuleNotFoundError:
                        data = _simple_yaml_load(bundle_dir.read_text(encoding="utf-8"))
                data.setdefault("project_dir", str(project_dir))
                return cls.model_validate(data).resolved(project_dir)
            raise FileNotFoundError(f"Unsupported design input file: {bundle_dir}")
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
                data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            except ModuleNotFoundError:
                data = _simple_yaml_load(manifest.read_text(encoding="utf-8"))
        data.setdefault("project_dir", str(bundle_dir))
        return cls.model_validate(data).resolved(bundle_dir)
