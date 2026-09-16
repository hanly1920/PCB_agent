from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ..schemas import ToolResult
from .command import CommandRunner


class KicadBridge:
    name = "kicad_bridge"

    def __init__(self) -> None:
        self.runner = CommandRunner()

    def apply_placement(
        self,
        board_path: str | Path,
        placements: dict[str, dict[str, Any]],
        output_board_path: str | Path,
    ) -> ToolResult:
        try:
            src = Path(board_path)
            dst = Path(output_board_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = src.read_text(encoding="utf-8", errors="replace")
            updated, missing = self._rewrite_footprints(text, placements)
            dst.write_text(updated, encoding="utf-8")
            backup = dst.with_suffix(dst.suffix + ".source.bak")
            shutil.copy2(src, backup)
            return ToolResult.success(
                self.name,
                f"Updated {len(placements) - len(missing)} footprints",
                artifacts={"board_path": str(dst), "backup_path": str(backup)},
                metrics={
                    "requested": len(placements),
                    "updated": len(placements) - len(missing),
                    "missing": len(missing),
                },
                observations=[f"Missing refs: {', '.join(missing)}"] if missing else [],
            )
        except Exception as exc:
            return ToolResult.failure(self.name, str(exc))

    @classmethod
    def _rewrite_footprints(
        cls, text: str, placements: dict[str, dict[str, Any]]
    ) -> tuple[str, list[str]]:
        spans = cls._top_level_blocks(text, {"footprint", "module"})
        replacements: list[tuple[int, int, str]] = []
        found: set[str] = set()
        for start, end in spans:
            block = text[start:end]
            ref = cls._extract_ref(block)
            if not ref or ref not in placements:
                continue
            p = placements[ref]
            new_block = cls._replace_top_level_at(
                block,
                float(p["x"]),
                float(p["y"]),
                float(p.get("rotation", 0.0)),
            )
            replacements.append((start, end, new_block))
            found.add(ref)
        for start, end, new in reversed(replacements):
            text = text[:start] + new + text[end:]
        return text, sorted(set(placements) - found)

    @staticmethod
    def _extract_ref(block: str) -> str | None:
        match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', block)
        if not match:
            match = re.search(r'\(fp_text\s+reference\s+"?([^"\s\)]+)"?', block)
        return match.group(1) if match else None

    @staticmethod
    def _replace_top_level_at(block: str, x: float, y: float, rot: float) -> str:
        i = 0
        depth = 0
        in_string = False
        escaped = False
        while i < len(block):
            ch = block[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == "(":
                if depth == 1 and block.startswith("(at", i):
                    j = i
                    nested = 0
                    nested_string = False
                    nested_escaped = False
                    while j < len(block):
                        current = block[j]
                        if nested_string:
                            if nested_escaped:
                                nested_escaped = False
                            elif current == "\\":
                                nested_escaped = True
                            elif current == '"':
                                nested_string = False
                        else:
                            if current == '"':
                                nested_string = True
                            elif current == "(":
                                nested += 1
                            elif current == ")":
                                nested -= 1
                                if nested == 0:
                                    replacement = f"(at {x:.6f} {y:.6f} {rot:.6f})"
                                    return block[:i] + replacement + block[j + 1 :]
                        j += 1
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        raise ValueError("Footprint has no top-level (at ...) field")

    @staticmethod
    def _top_level_blocks(text: str, heads: set[str]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        i = 0
        in_string = False
        escaped = False
        while i < len(text):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == "(":
                match = re.match(r"\(([A-Za-z_]+)", text[i:])
                if match and match.group(1) in heads:
                    start = i
                    depth = 0
                    j = i
                    nested_string = False
                    nested_escaped = False
                    while j < len(text):
                        current = text[j]
                        if nested_string:
                            if nested_escaped:
                                nested_escaped = False
                            elif current == "\\":
                                nested_escaped = True
                            elif current == '"':
                                nested_string = False
                        else:
                            if current == '"':
                                nested_string = True
                            elif current == "(":
                                depth += 1
                            elif current == ")":
                                depth -= 1
                                if depth == 0:
                                    out.append((start, j + 1))
                                    i = j
                                    break
                        j += 1
            i += 1
        return out
