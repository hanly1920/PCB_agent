from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pcbplace.dataset import task_from_json
from pcbplace.env import PlacementEnv
from pcbplace.runtime_safety import RUNTIME_PRIOR_SCHEMA_VERSION, RUNTIME_SHAPE_HINT_SCHEMA_VERSION


def _payload_with_module() -> dict:
    return {
        "board": {"bbox_mm": [0.0, 0.0, 20.0, 20.0], "grid_mm": 5.0},
        "components": [
            {
                "ref": "U1",
                "type": "IC",
                "size_mm": [2.0, 2.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
            },
            {
                "ref": "C1",
                "type": "C",
                "size_mm": [1.0, 1.0],
                "pads": [{"net": "N1", "rel_mm": [0.0, 0.0]}],
            },
        ],
        "nets": {"N1": ["U1", "C1"]},
        "graph": {"sequence": ["U1", "C1"]},
        "modules": [
            {
                "module_id": "M1",
                "module_type": "mixed",
                "anchor_ref": "U1",
                "members": ["U1", "C1"],
                "prior_region": {
                    "bbox_mm": [2.0, 2.0, 18.0, 18.0],
                    "confidence": 0.5,
                    "source": "rule_v1_generic_cluster",
                    "schema_version": RUNTIME_PRIOR_SCHEMA_VERSION,
                    "leakage_safe": True,
                },
                "shape_hint": {
                    "version": 2,
                    "schema_version": RUNTIME_SHAPE_HINT_SCHEMA_VERSION,
                    "source": "prior_rule_v1_generic_cluster",
                    "leakage_safe": True,
                    "module_center_mm": [10.0, 10.0],
                    "module_axis": "compact",
                    "anchor_relative_zone": "around",
                    "edge_corridor": "",
                    "support_ring_radius_mm": 4.0,
                },
            }
        ],
    }


class RuntimePriorHintWhitelistTest(unittest.TestCase):
    def _write(self, payload: dict, tmpdir: str, name: str) -> str:
        path = Path(tmpdir) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_whitelisted_runtime_prior_and_hint_load(self):
        with tempfile.TemporaryDirectory() as td:
            task = task_from_json(self._write(_payload_with_module(), td, "safe.json"), sequence_policy="stored")
            env = PlacementEnv(task)
            self.assertIsNotNone(env._module_region_bbox_for_ref("U1"))
            self.assertTrue(env._module_shape_hint_for_ref("U1"))

    def test_prior_without_schema_version_is_rejected(self):
        payload = _payload_with_module()
        payload["modules"][0]["prior_region"].pop("schema_version")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "prior_region.*schema_version"):
                task_from_json(self._write(payload, td, "bad_prior.json"), sequence_policy="stored")

    def test_shape_hint_without_leakage_safe_is_rejected(self):
        payload = _payload_with_module()
        payload["modules"][0]["shape_hint"].pop("leakage_safe")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "shape_hint.*leakage_safe"):
                task_from_json(self._write(payload, td, "bad_hint.json"), sequence_policy="stored")

    def test_legacy_region_bbox_without_prior_is_rejected(self):
        payload = _payload_with_module()
        module = payload["modules"][0]
        module.pop("prior_region")
        module["region_bbox_mm"] = [2.0, 2.0, 18.0, 18.0]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "legacy runtime prior"):
                task_from_json(self._write(payload, td, "legacy.json"), sequence_policy="stored")


if __name__ == "__main__":
    unittest.main()
