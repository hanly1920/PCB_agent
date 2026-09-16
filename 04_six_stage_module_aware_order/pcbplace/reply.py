from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .infer import infer_layout
try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover - exercised only when service deps are absent
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]


def _require_fastapi() -> None:
    if FastAPI is None or HTTPException is None:
        raise RuntimeError(
            "Reply service dependencies are not installed. "
            "Install them with: pip install fastapi uvicorn"
        )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    return int(value)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _resolve_ckpt_path(payload: Dict[str, Any], default_ckpt_path: Optional[str]) -> str:
    ckpt_path = payload.get("ckpt_path") or default_ckpt_path or os.environ.get("PCBPLACE_CKPT")
    if not ckpt_path:
        raise ValueError("Missing ckpt_path. Provide it in the request, --ckpt, or PCBPLACE_CKPT.")
    return str(ckpt_path)


def run_reply(
    payload: Dict[str, Any],
    *,
    default_ckpt_path: Optional[str] = None,
    default_device: str = "cuda",
) -> Dict[str, Any]:
    """Run one placement inference request.

    Accepted payload fields:
      - task_json_path: path to an existing structured board JSON, or
      - task: inline structured board JSON object
      - ckpt_path: optional checkpoint path; can also be supplied by server default
      - device, infer_objective_alpha, infer_region_alpha, beam_width, beam_topk,
        return_metrics, max_tokens, strict_state_dict
      - omitted infer_objective_alpha/infer_region_alpha restore checkpoint
        action_scoring metadata; supplied values explicitly override it
      - postprocess: whether to run layout post-processing after rollout
      - layout_preset: inference objective preset name, default "checkpoint"
      - objective_overrides: optional JSON object of PlacementEnv objective overrides
      - sequence_policy: checkpoint, rebuild, stored, or validate; default checkpoint

    Exactly one of task_json_path or task must be provided.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object.")

    task_json_path = payload.get("task_json_path")
    inline_task = payload.get("task")
    if bool(task_json_path) == bool(inline_task):
        raise ValueError("Provide exactly one of task_json_path or task.")

    ckpt_path = _resolve_ckpt_path(payload, default_ckpt_path)
    device = str(payload.get("device") or default_device or "cuda")
    infer_objective_alpha = _coerce_optional_float(payload.get("infer_objective_alpha"))
    infer_region_alpha = _coerce_optional_float(payload.get("infer_region_alpha"))
    beam_width = int(payload.get("beam_width", 1))
    beam_topk = int(payload.get("beam_topk", 16))
    return_metrics = _coerce_bool(payload.get("return_metrics"), True)
    max_tokens = _coerce_int(payload.get("max_tokens"), None)
    strict_state_dict = _coerce_bool(payload.get("strict_state_dict"), True)
    postprocess = _coerce_bool(payload.get("postprocess"), True)
    layout_preset = str(payload.get("layout_preset") or "checkpoint").strip() or "checkpoint"
    sequence_policy_raw = payload.get("sequence_policy")
    sequence_policy = (
        None
        if sequence_policy_raw is None
        or not str(sequence_policy_raw).strip()
        else str(sequence_policy_raw).strip()
    )

    objective_overrides_raw = payload.get("objective_overrides")
    if objective_overrides_raw is None:
        objective_overrides: Optional[Dict[str, Any]] = None
    elif isinstance(objective_overrides_raw, dict):
        objective_overrides = dict(objective_overrides_raw)
    else:
        raise ValueError("objective_overrides must be a JSON object or null.")

    temp_path: Optional[str] = None
    try:
        if inline_task is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
                json.dump(inline_task, f, ensure_ascii=False)
                temp_path = f.name
            task_path = temp_path
        else:
            task_path = str(task_json_path)
            if not Path(task_path).exists():
                raise ValueError(f"task_json_path does not exist: {task_path}")

        return infer_layout(
            task_path,
            ckpt_path,
            device=device,
            infer_objective_alpha=infer_objective_alpha,
            infer_region_alpha=infer_region_alpha,
            beam_width=beam_width,
            beam_topk=beam_topk,
            return_metrics=return_metrics,
            max_tokens=max_tokens,
            strict_state_dict=strict_state_dict,
            postprocess=postprocess,
            layout_preset=layout_preset,
            objective_overrides=objective_overrides,
            sequence_policy=sequence_policy,
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def create_app(
    *,
    default_ckpt_path: Optional[str] = None,
    default_device: str = "cuda",
):
    """Create the FastAPI reply service app."""
    _require_fastapi()
    app = FastAPI(
        title="pcb_autoplace reply service",
        version="1.0.0",
        description="HTTP wrapper around pcbplace.infer.infer_layout().",
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True}

    @app.post("/reply")
    def reply(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = run_reply(
                payload,
                default_ckpt_path=default_ckpt_path,
                default_device=default_device,
            )
            return {"ok": True, "result": result}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve pcb_autoplace inference via POST /reply.")
    parser.add_argument("--ckpt", default=None, help="Default checkpoint path. Can also be set with PCBPLACE_CKPT.")
    parser.add_argument("--device", default="cuda", help="Torch device for inference. infer_layout is CUDA-only by default.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    _require_fastapi()
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is not installed. Install it with: pip install uvicorn") from exc

    app = create_app(default_ckpt_path=args.ckpt, default_device=args.device)
    uvicorn.run(app, host=args.host, port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
