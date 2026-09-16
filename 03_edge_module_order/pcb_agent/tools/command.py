from __future__ import annotations
import subprocess, time
from pathlib import Path
from typing import Sequence
from ..schemas.tool_result import ToolResult

class CommandRunner:
    def run(self, name: str, args: Sequence[str], *, cwd: str | Path | None = None, timeout_sec: int = 600, allowed_returncodes: set[int] | None = None) -> ToolResult:
        allowed = allowed_returncodes or {0}
        started = time.time()
        try:
            proc = subprocess.run(list(map(str, args)), cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_sec, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult.failure(name, str(exc), raw={"command": list(map(str, args)), "elapsed_sec": time.time()-started})
        raw = {"command": list(map(str, args)), "cwd": str(cwd) if cwd else None, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "elapsed_sec": time.time()-started}
        if proc.returncode not in allowed:
            return ToolResult.failure(name, f"Command exited with {proc.returncode}", raw=raw)
        return ToolResult.success(name, f"Command completed with {proc.returncode}", raw=raw)
