from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.env_cuda import build_context_tokens_cuda


def _toy_task(expert_shift: float = 0.0) -> dict:
    return {
        "board": {"bbox_mm": [0.0, 0.0, 20.0, 20.0], "grid_mm": 5.0},
        "components": [
            {
                "ref": "U1",
                "type": "IC",
                "size_mm": [2.0, 2.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
                "expert": {"xy_mm": [5.0 + expert_shift, 5.0], "rot": 0},
            },
            {
                "ref": "C1",
                "type": "C",
                "size_mm": [1.0, 1.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
                "anchor_ref": "U1",
                "expert": {"xy_mm": [15.0 - expert_shift, 15.0], "rot": 90},
            },
        ],
        "nets": {"N1": ["U1", "C1"]},
        "graph": {"sequence": ["U1", "C1"]},
    }


class ExpertLeakageGuardTest(unittest.TestCase):
    def _write(self, payload: dict, tmpdir: str, name: str) -> str:
        path = Path(tmpdir) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_task_from_json_strips_expert_fields_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(_toy_task(), td, "task.json")
            task = task_from_json(path, sequence_policy="stored")
            self.assertTrue(all(c.expert_xy is None for c in task.components))
            self.assertTrue(all(c.expert_rot == 0 for c in task.components))
            self.assertFalse((task.sequence_meta or {}).get("expert_fields_loaded"))

    def test_context_tokens_invariant_to_expert_coordinates_when_stripped(self):
        with tempfile.TemporaryDirectory() as td:
            path_a = self._write(_toy_task(0.0), td, "a.json")
            path_b = self._write(_toy_task(7.0), td, "b.json")
            env_a = PlacementEnv(task_from_json(path_a, sequence_policy="stored", load_expert=False))
            env_b = PlacementEnv(task_from_json(path_b, sequence_policy="stored", load_expert=False))
            tok_a = build_context_tokens_cuda(env_a, env_a.current_ref(), torch.device("cpu"), max_tokens=8)
            tok_b = build_context_tokens_cuda(env_b, env_b.current_ref(), torch.device("cpu"), max_tokens=8)
            self.assertTrue(torch.equal(tok_a, tok_b))

    def test_runtime_env_rejects_loaded_expert_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(_toy_task(), td, "task.json")
            task = task_from_json(path, sequence_policy="stored", load_expert=True)
            with self.assertRaises(ValueError):
                PlacementEnv(task)


if __name__ == "__main__":
    unittest.main()
