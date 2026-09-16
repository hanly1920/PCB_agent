from __future__ import annotations
import json
import urllib.request
from typing import Any
from ..config import LLMConfig

class QwenClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> str:
        if self.config.provider in {"mock", "disabled"}:
            raise RuntimeError(f"LLM provider {self.config.provider!r} does not make network calls")
        endpoint = self.config.endpoint.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.provider == "ollama":
            url = endpoint + "/api/chat"
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config.temperature},
                "format": "json",
            }
        else:
            url = endpoint + "/v1/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if json_schema:
                payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "layout_dsl", "schema": json_schema}}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if self.config.provider == "ollama":
            return str((data.get("message") or {}).get("content") or "")
        return str(data["choices"][0]["message"]["content"])
