from __future__ import annotations
from pathlib import Path
from typing import Any
from ..agent import AgentOrchestrator
from ..config import AgentConfig

def create_app(config:AgentConfig|None=None):
    try:from fastapi import FastAPI,HTTPException
    except ModuleNotFoundError as exc:raise RuntimeError("Install fastapi and uvicorn for the API service") from exc
    cfg=config or AgentConfig();app=FastAPI(title="PCB Agent",version="0.1.0")
    @app.get("/healthz")
    def healthz()->dict[str,bool]:return {"ok":True}
    @app.post("/run")
    def run(payload:dict[str,Any])->dict[str,Any]:
        try:
            state=AgentOrchestrator(cfg).run(payload["bundle"],payload["out"],payload.get("text", ""));return state.model_dump(mode="json")
        except Exception as exc:raise HTTPException(status_code=500,detail=str(exc)) from exc
    return app
