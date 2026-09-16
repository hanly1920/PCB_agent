from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_json
from .env import Component, Task
from .json_schema import validate_task_json

def task_from_json(path: str) -> Task:
    data = load_json(path)
    validate_task_json(data, source=path)
    board = data["board"]
    bbox = tuple(board["bbox_mm"])
    grid = float(board.get("grid_mm", 1.0))
    comps: List[Component] = []
    for c in data["components"]:
        pads = [(p["net"], tuple(p["rel_mm"])) for p in c.get("pads", [])]
        review = c.get("semantic_review") or {}
        comps.append(Component(
            ref=c["ref"],
            type=c.get("type","misc"),
            size_mm=tuple(c.get("size_mm",[1.0,1.0])),
            pads=pads,
            allowed_sides=c.get("allowed_sides", []) or [],
            must_touch_boundary=(
                c.get("must_touch_boundary")
                if c.get("must_touch_boundary", None) is not None
                else (c.get("semantic") or {}).get("must_touch_boundary")
            ),
            semantic_class=(c.get("semantic_class") or (c.get("semantic") or {}).get("semantic_class") or "other"),
            region_type=(c.get("region_type") or (c.get("semantic") or {}).get("region_type") or "free"),
            functional_group=(c.get("functional_group") or (c.get("semantic") or {}).get("functional_group") or "misc"),
            side_preference=(c.get("side_preference") or (c.get("semantic") or {}).get("side_preference") or "free"),
            align_group=(c.get("align_group") or (c.get("semantic") or {}).get("align_group")),
            anchor_ref=(c.get("anchor_ref") or (c.get("semantic") or {}).get("anchor_ref") or None),
            subzone=(c.get("subzone") or (c.get("semantic") or {}).get("subzone") or "free"),
            same_side_group=(c.get("same_side_group") or (c.get("semantic") or {}).get("same_side_group") or None),
            boundary_order=(
                int(c.get("boundary_order")) if c.get("boundary_order", None) not in (None, "")
                else (int((c.get("semantic") or {}).get("boundary_order")) if (c.get("semantic") or {}).get("boundary_order", None) not in (None, "") else None)
            ),
            critical_nets=tuple(c.get("critical_nets") or (c.get("semantic") or {}).get("critical_nets") or []),
            critical_neighbors=tuple(c.get("critical_neighbors") or (c.get("semantic") or {}).get("critical_neighbors") or []),
            review_status=str(review.get("review_status") or "untracked"),
            auto_confidence=float(review.get("auto_confidence", 1.0) if review.get("auto_confidence", None) not in (None, "") else 1.0),
            needs_review=bool(review.get("needs_review", False)),
        ))
    seq = data.get("graph", {}).get("sequence", [])
    nets = data.get("nets", {}) if isinstance(data.get("nets", {}), dict) else {}
    return Task(bbox_mm=bbox, grid_mm=grid, components=comps, nets=nets, sequence=seq)

@dataclass
class Traj:
    obs_tokens: np.ndarray  # [T, D]
    actions: np.ndarray     # [T] flattened action index
    masks: np.ndarray       # [T, A] 0/1 legal mask
    bias: np.ndarray        # [T, A] bias

class TrajectoryDataset(Dataset):
    """Build expert trajectories from json by re-discretizing the expert placement positions if provided.
    If json doesn't contain expert positions, this dataset cannot be constructed (by design).
    Expected json extension:
      components[i]["expert"]= {"xy_mm":[x,y], "rot":0}
    """
    def __init__(self, json_paths: List[str]):
        self.paths = json_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = load_json(self.paths[idx])
        # Build task
        task = task_from_json(self.paths[idx])
        # Build expert action sequence from provided expert placements
        # For clean pipeline, we require expert placements for training set.
        placements = {}
        for c in data["components"]:
            if "expert" in c:
                placements[c["ref"]] = (float(c["expert"]["xy_mm"][0]), float(c["expert"]["xy_mm"][1]), int(c["expert"].get("rot",0)))
        if len(placements) != len(task.sequence):
            raise ValueError(f"Missing expert placements for some refs in {self.paths[idx]}")
        return {"task_path": self.paths[idx]}
