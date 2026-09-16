from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from env import PCBPlacementEnv
from reward_utils import (
    DEFAULT_SLW_PENALTY_WEIGHT,
    returns_from_rewards,
    step_reward,
    trajectory_score,
)


@dataclass
class TrajectoryTensors:
    """Container for a fully processed expert trajectory."""

    states: torch.FloatTensor  # (T, 4, N, N)
    actions: torch.LongTensor  # (T, 3)
    rewards: torch.FloatTensor  # (T, 1)
    returns_to_go: torch.FloatTensor  # (T, 1)
    timesteps: torch.LongTensor  # (T,)
    hpwl: torch.FloatTensor  # (T,1)
    slw: torch.FloatTensor  # (T,1)
    nslw: torch.FloatTensor  # (T,1)
    score: torch.FloatTensor  # (T,1)
    hi: torch.FloatTensor  # (F,)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize_rotation(rot_deg: float) -> int:
    return int(round(rot_deg / 90.0)) % 4


class ExpertTrajectoryProcessor:
    """Processes a single expert trajectory folder into env-ready tensors."""

    HI_FEATURE_DIM = 11
    SCORE_LAMBDA1 = 1000.0
    SCORE_LAMBDA2 = 0.01

    def __init__(
        self,
        trajectory_dir: Path,
        grid_size: Optional[int] = None,
        margin: int = 2,
        placement_mode: str = "heuristic",  # 'heuristic' preserves current search, 'layout' tries raw expert layout
    ) -> None:
        self.trajectory_dir = trajectory_dir
        self.fixed_grid_size = grid_size
        self.margin = margin
        if placement_mode not in {"heuristic", "layout"}:
            raise ValueError("placement_mode must be 'heuristic' or 'layout'")
        self.placement_mode = placement_mode

        base_name = trajectory_dir.name.replace("_traj", "")
        task_path = trajectory_dir / f"{base_name}_task.json"
        layout_path = trajectory_dir / f"{base_name}_layout.json"

        if not task_path.exists() or not layout_path.exists():
            raise FileNotFoundError(
                f"Trajectory folder {trajectory_dir} missing task/layout json files"
            )

        self.task_data = _load_json(task_path)
        self.layout_data = _load_json(layout_path)
        self._prepare_geometry()
        self._prepare_components()
        self._prepare_spacing_rules()
        self._prepare_netlist()
        self._prepare_action_map()

    # ------------------------------------------------------------------
    # Geometry preparation
    # ------------------------------------------------------------------
    def _prepare_geometry(self) -> None:
        boundary = self.task_data.get("boundary")
        if not boundary:
            raise ValueError(f"Task file {self.trajectory_dir} missing boundary polygon")

        xs = [point[0] for point in boundary]
        ys = [point[1] for point in boundary]

        self._world_min_x = min(xs)
        self._world_min_y = min(ys)
        world_width = max(xs) - self._world_min_x
        world_height = max(ys) - self._world_min_y

        if world_width <= 0 or world_height <= 0:
            raise ValueError(f"Invalid boundary dimensions in {self.trajectory_dir}")

        # Use fixed grid_size if provided, otherwise calculate based on board dimensions
        if self.fixed_grid_size is not None:
            self.grid_size = self.fixed_grid_size
        else:
            # Calculate appropriate grid size based on board dimensions
            # Aim for ~1mm per grid cell resolution
            target_resolution = 1.0  # mm per cell
            grid_width = max(32, int(round(world_width / target_resolution)))
            grid_height = max(32, int(round(world_height / target_resolution)))
            self.grid_size = max(grid_width, grid_height)
            
            # Ensure grid_size is reasonable (not too large)
            self.grid_size = min(self.grid_size, 256)

        usable_grid = self.grid_size - 2 * self.margin
        if usable_grid <= 0:
            raise ValueError("Grid size too small for configured margin")

        scale = usable_grid / max(world_width, world_height)
        self._scale = scale

        board_width_grid = world_width * scale
        board_height_grid = world_height * scale

        remaining_x = self.grid_size - board_width_grid
        remaining_y = self.grid_size - board_height_grid

        self._offset_x = remaining_x / 2.0
        self._offset_y = remaining_y / 2.0

        polygon_grid: List[Tuple[int, int]] = []
        for x, y in boundary:
            gx = round((x - self._world_min_x) * self._scale + self._offset_x)
            gy = round((y - self._world_min_y) * self._scale + self._offset_y)
            polygon_grid.append((int(gx), int(gy)))

        self.boundary_polygon = polygon_grid
        xs_grid = [p[0] for p in polygon_grid]
        ys_grid = [p[1] for p in polygon_grid]
        self._boundary_bbox = (
            min(xs_grid),
            min(ys_grid),
            max(xs_grid),
            max(ys_grid),
        )

    # ------------------------------------------------------------------
    # Component and netlist preparation
    # ------------------------------------------------------------------
    def _prepare_components(self) -> None:
        components_raw: Sequence[Dict] = self.task_data.get("components", [])
        if not components_raw:
            raise ValueError(f"No components listed in task file {self.trajectory_dir}")

        layout_names = set(self.layout_data.keys())
        filtered: List[Dict] = []
        seen: set[str] = set()

        for comp in components_raw:
            name = comp.get("name")
            if not name or name not in layout_names:
                continue
            if name in seen:
                continue
            seen.add(name)
            filtered.append(comp)

        if not filtered:
            raise ValueError(
                f"No components shared between task/layout for {self.trajectory_dir}"
            )

        self.type_to_id: Dict[str, int] = {}
        component_list: List[Dict] = []
        name_to_id: Dict[str, int] = {}
        max_pads = 0

        for comp_id, comp in enumerate(filtered):
            name = comp["name"]
            type_name = comp.get("type", "default")
            type_id = self.type_to_id.setdefault(type_name, len(self.type_to_id))

            width_cells = max(1, int(round(comp.get("w", 1.0) * self._scale)))
            height_cells = max(1, int(round(comp.get("h", 1.0) * self._scale)))

            pad_list_raw: Sequence[Sequence[float]] = comp.get("pads", [])
            pad_list: List[List[int]] = []
            for pad in pad_list_raw:
                pad_x = int(round(pad[0] * self._scale))
                pad_y = int(round(pad[1] * self._scale))
                pad_x = min(max(pad_x, 0), max(width_cells - 1, 0))
                pad_y = min(max(pad_y, 0), max(height_cells - 1, 0))
                pad_list.append([pad_x, pad_y])

            max_pads = max(max_pads, len(pad_list))

            component_list.append(
                {
                    "comp_id": comp_id,
                    "name": name,
                    "type_id": type_id,
                    "size": [width_cells, height_cells],
                    "pad_list": pad_list,
                }
            )
            name_to_id[name] = comp_id

        self.component_list = component_list
        self._name_to_id = name_to_id
        self.max_pads = max_pads if max_pads > 0 else 1
        self.num_component_types = len(self.type_to_id)

    def _prepare_netlist(self) -> None:
        netlist_raw: Sequence = self.task_data.get("nets", [])
        netlist: List[List[Tuple[int, int]]] = []

        for connection in netlist_raw:
            processed: List[Tuple[int, int]] = []
            for entry in connection:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                comp_name, pad_idx = entry
                comp_id = self._name_to_id.get(comp_name)
                if comp_id is None:
                    continue
                processed.append((comp_id, int(pad_idx)))
            if len(processed) >= 2:
                netlist.append(processed)

        self.netlist = netlist

    def _prepare_spacing_rules(self) -> None:
        type_ids = list(range(self.num_component_types))

        def to_grid_spacing(value: float) -> int:
            value = max(0.0, float(value))
            if value == 0.0:
                return 0
            return int(math.ceil(value * self._scale))

        default_body_mm = self.task_data.get("min_spacing", 0.0)
        default_pad_mm = self.task_data.get("min_pad_spacing", default_body_mm)

        default_body = to_grid_spacing(default_body_mm)
        default_pad = to_grid_spacing(default_pad_mm)

        body_rules: Dict[Tuple[int, int], int] = {}
        pad_rules: Dict[Tuple[int, int], int] = {}

        for t1 in type_ids:
            for t2 in type_ids:
                body_rules[(t1, t2)] = default_body
                pad_rules[(t1, t2)] = default_pad

        raw_body = self.task_data.get("body_spacing_rules", {}) or {}
        for type_a, mapping in raw_body.items():
            type_a_id = self.type_to_id.get(type_a)
            if type_a_id is None:
                continue
            for type_b, distance in (mapping or {}).items():
                type_b_id = self.type_to_id.get(type_b)
                if type_b_id is None:
                    continue
                grid_dist = to_grid_spacing(distance)
                body_rules[(type_a_id, type_b_id)] = grid_dist
                body_rules[(type_b_id, type_a_id)] = grid_dist

        raw_pad = self.task_data.get("pad_spacing_rules", {}) or {}
        for type_a, mapping in raw_pad.items():
            type_a_id = self.type_to_id.get(type_a)
            if type_a_id is None:
                continue
            for type_b, distance in (mapping or {}).items():
                type_b_id = self.type_to_id.get(type_b)
                if type_b_id is None:
                    continue
                grid_dist = to_grid_spacing(distance)
                pad_rules[(type_a_id, type_b_id)] = grid_dist
                pad_rules[(type_b_id, type_a_id)] = grid_dist

        self.body_spacing_rules = body_rules
        self.pad_spacing_rules = pad_rules

    def _candidate_axis_positions(
        self,
        center: float,
        span: int,
        bound_min: int,
        bound_max: int,
    ) -> List[int]:
        raw_start = center - span / 2.0
        candidates: List[Tuple[float, int]] = []
        for pos in range(bound_min, bound_max - span + 1):
            candidate_center = pos + span / 2.0
            distance = abs(candidate_center - center)
            candidates.append((distance, pos))

        candidates.sort(key=lambda item: item[0])
        return [value for _, value in candidates]
    def _prepare_action_map(self) -> None:
        """Dispatch to placement strategy based on placement_mode."""
        if self.placement_mode == "layout":
            self._prepare_action_map_layout()
        else:
            self._prepare_action_map_heuristic()

    def _prepare_action_map_layout(self) -> None:
        """Direct layout reproduction: place components at expert-given (x,y,rot) anchors.

        Steps:
          1. Iterate env placement sequence (ordering consistency)
          2. Try raw top-left anchor (layout x,y) with provided rotation
          3. If blocked, spiral (Manhattan ring) outward search up to radius 12
          4. Fallback global scan (keep desired rot first, then others)
        Fails fast if still impossible.
        """
        env = self.build_env()
        placement_order = env.compute_placement_sequence(self.component_list)
        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self._boundary_bbox
        occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        action_map: Dict[int, Tuple[int, int, int]] = {}

        def in_bbox(x: int, y: int, w: int, h: int) -> bool:
            # Allow slight overflow to handle scaling issues
            margin = 2
            return (x >= bbox_min_x - margin and 
                    y >= bbox_min_y - margin and 
                    (x + w) <= bbox_max_x + margin and 
                    (y + h) <= bbox_max_y + margin)

        for comp in placement_order:
            pose = self.layout_data.get(comp["name"])
            if pose is None:
                continue
            comp_id = comp["comp_id"]
            desired_rot = _normalize_rotation(pose.get("rot", 0))
            base_w, base_h = comp["size"]
            if desired_rot % 2 == 1:
                w, h = base_h, base_w
            else:
                w, h = base_w, base_h

            anchor_x, anchor_y = self._world_to_grid(pose["x"], pose["y"])
            x0 = int(round(anchor_x))
            y0 = int(round(anchor_y))

            placed = False
            # Try raw
            if in_bbox(x0, y0, w, h) and env._check_boundary(x0, y0, w, h):
                if not np.any(occupancy[y0:y0+h, x0:x0+w]):
                    occupancy[y0:y0+h, x0:x0+w] = 1
                    action_map[comp_id] = (x0, y0, desired_rot)
                    placed = True

            # Spiral local search
            if not placed:
                max_radius = 12
                candidates = []
                for dx in range(-max_radius, max_radius + 1):
                    for dy in range(-max_radius, max_radius + 1):
                        if dx == 0 and dy == 0:  # Skip center, already tried
                            continue
                        dist = abs(dx) + abs(dy)  # Manhattan distance
                        if dist > max_radius:
                            continue
                        candidates.append((dist, dx, dy))
                
                # Sort by distance
                candidates.sort(key=lambda x: x[0])
                
                for dist, dx, dy in candidates:
                    x = x0 + dx
                    y = y0 + dy
                    if not in_bbox(x, y, w, h):
                        continue
                    if not env._check_boundary(x, y, w, h):
                        continue
                    if np.any(occupancy[y:y+h, x:x+w]):
                        continue
                    occupancy[y:y+h, x:x+w] = 1
                    action_map[comp_id] = (x, y, desired_rot)
                    placed = True
                    break

            # Global fallback scan (desired rotation then other rotations)
            if not placed:
                rot_order = [desired_rot] + [r for r in range(4) if r != desired_rot]
                for rot in rot_order:
                    if rot % 2 == 1:
                        w2, h2 = base_h, base_w
                    else:
                        w2, h2 = base_w, base_h
                    # Scan entire grid, not just bbox
                    for y in range(self.grid_size - h2 + 1):
                        if placed:
                            break
                        for x in range(self.grid_size - w2 + 1):
                            if not env._check_boundary(x, y, w2, h2):
                                continue
                            if np.any(occupancy[y:y+h2, x:x+w2]):
                                continue
                            occupancy[y:y+h2, x:x+w2] = 1
                            action_map[comp_id] = (x, y, rot)
                            placed = True
                            break
                    if placed:
                        break

            if not placed:
                # Fallback to heuristic placement if layout fails
                print(f"Layout placement failed for component {comp['name']} in {self.trajectory_dir}, falling back to heuristic")
                # Use heuristic for this component
                pose = self.layout_data.get(comp["name"])
                desired_rot = _normalize_rotation(pose.get("rot", 0)) if pose else 0
                rotation_candidates = [desired_rot] + [r for r in range(4) if r != desired_rot]

                anchor_x, anchor_y = self._world_to_grid(pose["x"], pose["y"]) if pose else (self.grid_size // 2, self.grid_size // 2)

                best_distance = float("inf")
                best_action: Optional[Tuple[int,int,int]] = None

                dyn_radius = max(8, max(base_w, base_h) // 2 + 4)

                for rot_idx in rotation_candidates:
                    if rot_idx % 2 == 1:
                        w3, h3 = base_h, base_w
                    else:
                        w3, h3 = base_w, base_h

                    x0 = int(round(anchor_x))
                    y0 = int(round(anchor_y))

                    for radius in range(1, dyn_radius + 1):
                        for dx in range(-radius, radius + 1):
                            dy_candidates = [radius - abs(dx), -(radius - abs(dx))]
                            for dy in dy_candidates:
                                x = x0 + dx
                                y = y0 + dy
                                if x < 0 or y < 0 or x + w3 > self.grid_size or y + h3 > self.grid_size:
                                    continue
                                if not env._check_boundary(x, y, w3, h3):
                                    continue
                                region = occupancy[y:y+h3, x:x+w3]
                                if np.any(region):
                                    continue
                                dx_c = x - x0
                                dy_c = y - y0
                                dist = dx_c*dx_c + dy_c*dy_c
                                if dist < best_distance:
                                    best_distance = dist
                                    best_action = (x, y, rot_idx)

                if best_action is not None:
                    x, y, rot_idx = best_action
                    if rot_idx % 2 == 1:
                        w3, h3 = base_h, base_w
                    else:
                        w3, h3 = base_w, base_h
                    occupancy[y:y+h3, x:x+w3] = 1
                    action_map[comp_id] = (x, y, rot_idx)
                    placed = True

            if not placed:
                raise ValueError(
                    f"Both layout and heuristic placement failed for component {comp['name']} in {self.trajectory_dir}"
                )

        self.action_map = action_map

    def _prepare_action_map_heuristic(self) -> None:
        """Heuristic local search (previous default implementation)."""
        env = self.build_env()
        placement_order = env.compute_placement_sequence(self.component_list)

        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self._boundary_bbox
        occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        action_map: Dict[int, Tuple[int, int, int]] = {}

        local_radius = 8  # 可调：局部搜索半径

        for comp in placement_order:
            pose = self.layout_data.get(comp["name"])
            if pose is None:
                continue
            comp_id = comp["comp_id"]
            desired_rot = _normalize_rotation(pose.get("rot", 0))
            rotation_candidates = [desired_rot] + [r for r in range(4) if r != desired_rot]

            anchor_x, anchor_y = self._world_to_grid(pose["x"], pose["y"])

            placed = False
            best_distance = float("inf")
            best_action: Optional[Tuple[int,int,int]] = None

            dyn_radius = max(local_radius, max(comp["size"]) // 2 + 4)

            for rot_idx in rotation_candidates:
                base_w, base_h = comp["size"]
                if rot_idx % 2 == 1:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h

                x0 = int(round(anchor_x))
                y0 = int(round(anchor_y))

                def in_bbox(x: int, y: int) -> bool:
                    return x >= bbox_min_x and y >= bbox_min_y and (x + w) <= bbox_max_x and (y + h) <= bbox_max_y

                if in_bbox(x0, y0) and env._check_boundary(x0, y0, w, h):
                    region0 = occupancy[y0:y0+h, x0:x0+w]
                    if not np.any(region0):
                        distance0 = 0.0
                        if distance0 < best_distance:
                            best_distance = distance0
                            best_action = (x0, y0, rot_idx)

                for radius in range(1, dyn_radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if abs(dx) + abs(dy) != radius:  # Only Manhattan distance == radius
                                continue
                            x = x0 + dx
                            y = y0 + dy
                            if not in_bbox(x, y):
                                continue
                            if not env._check_boundary(x, y, w, h):
                                continue
                            region = occupancy[y:y+h, x:x+w]
                            if np.any(region):
                                continue
                            dx_c = x - x0
                            dy_c = y - y0
                            dist = dx_c*dx_c + dy_c*dy_c
                            if dist < best_distance:
                                best_distance = dist
                                best_action = (x, y, rot_idx)

            if best_action is not None:
                x, y, rot_idx = best_action
                base_w, base_h = comp["size"]
                if rot_idx % 2 == 1:
                    w, h = base_h, base_w
                else:
                    w, h = base_w, base_h
                occupancy[y:y+h, x:x+w] = 1
                action_map[comp_id] = (x, y, rot_idx)
                placed = True
            else:
                for rot_idx in rotation_candidates:
                    base_w, base_h = comp["size"]
                    if rot_idx % 2 == 1:
                        w, h = base_h, base_w
                    else:
                        w, h = base_w, base_h
                    for y in range(bbox_min_y, bbox_max_y - h + 1):
                        if placed:
                            break
                        for x in range(bbox_min_x, bbox_max_x - w + 1):
                            if not env._check_boundary(x, y, w, h):
                                continue
                            region = occupancy[y:y+h, x:x+w]
                            if np.any(region):
                                continue
                            occupancy[y:y+h, x:x+w] = 1
                            action_map[comp_id] = (x, y, rot_idx)
                            placed = True
                            break
                    if placed:
                        break

            if not placed:
                raise ValueError(
                    f"Fast placement failed for component {comp['name']} in {self.trajectory_dir} "
                    f"(anchor=({anchor_x:.2f},{anchor_y:.2f})) 尝试旋转与局部/全局搜索均失败"
                )

        self.action_map = action_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        gx = (x - self._world_min_x) * self._scale + self._offset_x
        gy = (y - self._world_min_y) * self._scale + self._offset_y
        return gx, gy

    def build_env(self) -> PCBPlacementEnv:
        return PCBPlacementEnv(
            grid_size=self.grid_size,
            num_rotations=4,
            max_pads=self.max_pads,
            num_component_types=self.num_component_types,
            boundary_polygon=self.boundary_polygon,
            component_list=[comp.copy() for comp in self.component_list],
            netlist=[list(net) for net in self.netlist],
            body_spacing_rules=self.body_spacing_rules.copy(),
            pad_spacing_rules=self.pad_spacing_rules.copy(),
        )

    def _build_state_tensor(
        self,
        obs: Dict,
        rotation: Optional[int] = None,
        bonus_quantile: float = 0.25,
    ) -> torch.FloatTensor:
        """Project environment observation to 4-channel mask tensor expected by the model.

        Channels (0-indexed):
            0. view_mask         — occupied cells (float32 0/1)
            1. position_mask     — legality map with unified semantics (0 legal, 1 illegal)
            2. wire_mask         — normalized HPWL/SLW heuristic (lower is better)
            3. bonus_mask        — subset of legal cells with low wire cost (1 where bonus applies)

        When ``rotation`` is provided, the position/wire channels are built from the
        corresponding rotation slice; otherwise a rotation-agnostic aggregation is used,
        enabling policy rollouts prior to sampling the rotation.
        """

        view_mask = obs["view_mask"].astype(np.float32)
        position_mask = obs["position_mask"].astype(np.float32)
        wire_mask = obs["wire_mask"].astype(np.float32)

        rot_idx: Optional[int] = None
        if rotation is not None and position_mask.ndim == 3:
            rot_idx = int(rotation) % position_mask.shape[0]

        if rot_idx is not None:
            legal_map = (position_mask[rot_idx] == 0)
            wire_map = wire_mask[rot_idx]
        else:
            legal_map = (position_mask == 0).any(axis=0)
            wire_map = wire_mask.min(axis=0)

        # Channel 1: legality mask with unified semantics (0 legal, 1 illegal)
        position_channel = np.where(legal_map, 0.0, 1.0).astype(np.float32)

        # Channel 2: normalized wire cost heuristic
        wire_map = wire_map.astype(np.float32)
        if np.isfinite(wire_map).any():
            denom = float(np.max(np.abs(wire_map)))
            if denom > 1e-6:
                wire_channel = wire_map / denom
            else:
                wire_channel = np.zeros_like(wire_map, dtype=np.float32)
        else:
            wire_channel = np.zeros_like(wire_map, dtype=np.float32)

        # Channel 3: bonus mask highlighting low-cost legal placements
        bonus_mask = np.zeros_like(wire_channel, dtype=np.float32)
        legal_indices = legal_map & np.isfinite(wire_map)
        if legal_indices.any():
            legal_values = wire_map[legal_indices]
            try:
                threshold = np.quantile(legal_values, bonus_quantile)
            except ValueError:
                threshold = legal_values.min()
            bonus_candidates = legal_indices & (wire_map <= threshold)
            bonus_mask[bonus_candidates] = 1.0
        # ensure bonuses only on legal cells
        bonus_mask = bonus_mask * (1.0 - position_channel)

        stacked = np.stack([view_mask, position_channel, wire_channel, bonus_mask], axis=0)
        return torch.from_numpy(stacked.copy())

    def _compute_hi_features(self) -> np.ndarray:
        comps = self.component_list
        comp_count = len(comps)
        widths = np.array([comp["size"][0] for comp in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)
        heights = np.array([comp["size"][1] for comp in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)
        areas = widths * heights
        pad_counts = np.array([len(comp["pad_list"]) for comp in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)

        total_area = float(np.sum(areas))
        grid_area = float(self.grid_size ** 2)
        density = total_area / grid_area if grid_area > 0 else 0.0
        avg_area = float(np.mean(areas)) if comp_count else 0.0
        std_area = float(np.std(areas)) if comp_count else 0.0
        max_area = float(np.max(areas)) if comp_count else 0.0

        avg_pad = float(np.mean(pad_counts)) if comp_count else 0.0
        std_pad = float(np.std(pad_counts)) if comp_count else 0.0
        max_pads = max(self.max_pads, 1)

        net_lengths = [len(net) for net in self.netlist]
        net_count = len(net_lengths)
        avg_net_degree = float(np.mean(net_lengths)) if net_lengths else 0.0

        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self._boundary_bbox
        board_w = float(max(bbox_max_x - bbox_min_x, 1))
        board_h = float(max(bbox_max_y - bbox_min_y, 1))
        aspect_ratio = board_w / board_h if board_h > 0 else 1.0

        features = np.array(
            [
                comp_count / 512.0,
                avg_area / grid_area if grid_area > 0 else 0.0,
                std_area / grid_area if grid_area > 0 else 0.0,
                max_area / grid_area if grid_area > 0 else 0.0,
                avg_pad / max_pads,
                std_pad / max_pads,
                net_count / 512.0,
                avg_net_degree / max_pads,
                density,
                aspect_ratio,
                self.grid_size / 256.0,
            ],
            dtype=np.float32,
        )

        return features

    def rollout(self) -> TrajectoryTensors:
        env = self.build_env()
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        states: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        rewards: List[float] = []
        timesteps: List[int] = []
        hpwls: List[float] = []
        slws: List[float] = []
        nslws: List[float] = []
        scores: List[float] = []

        prev_hpwl = float(getattr(env, "hpwl", 0.0))
        prev_slw = float(getattr(env, "slw", 0.0))

        for step_idx, component in enumerate(getattr(env, "placement_sequence", [])):
            comp_id = component["comp_id"]
            action = self.action_map.get(comp_id)
            if action is None:
                continue

            state_tensor = self._build_state_tensor(obs, action[2])
            states.append(state_tensor)
            actions.append(torch.tensor(action, dtype=torch.long))
            timesteps.append(step_idx)

            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                raise RuntimeError("Unexpected step result length")

            if info.get("illegal"):
                raise RuntimeError(
                    f"Illegal action ({info.get('reason')}) when replaying component {component['name']}"
                )

            # collect per-step info metrics if provided
            hpwl_curr = float(info.get('hpwl', float('nan')))
            slw_curr = float(info.get('slw', float('nan')))
            if math.isnan(hpwl_curr):
                hpwl_curr = prev_hpwl
            if math.isnan(slw_curr):
                slw_curr = prev_slw

            hpwls.append(hpwl_curr)
            slws.append(slw_curr)
            nslw_curr = float(info.get('nslw', float('nan')))
            if math.isnan(nslw_curr):
                nslw_curr = nslws[-1] if nslws else 0.0

            nslws.append(nslw_curr)

            reward_step = step_reward(
                prev_hpwl=prev_hpwl,
                curr_hpwl=hpwl_curr,
                prev_slw=prev_slw,
                curr_slw=slw_curr,
                hpwl_threshold=None,
                slw_penalty_weight=DEFAULT_SLW_PENALTY_WEIGHT,
            )
            rewards.append(reward_step)

            prev_hpwl = hpwl_curr
            prev_slw = slw_curr
            if done:
                break

        if not states:
            raise RuntimeError(f"Failed to generate trajectory for {self.trajectory_dir}")

        states_tensor = torch.stack(states, dim=0)
        actions_tensor = torch.stack(actions, dim=0)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

        hpwl_tensor = torch.tensor(hpwls, dtype=torch.float32).unsqueeze(-1)
        slw_tensor = torch.tensor(slws, dtype=torch.float32).unsqueeze(-1)
        nslw_tensor = torch.tensor(nslws, dtype=torch.float32).unsqueeze(-1)

        final_hpwl = hpwls[-1] if hpwls else prev_hpwl
        final_nslw = nslws[-1] if nslws else 0.0
        final_score = trajectory_score(
            self.SCORE_LAMBDA1,
            self.SCORE_LAMBDA2,
            hpwl_final=final_hpwl,
            nslw_final=final_nslw,
        )
        scores = [final_score for _ in rewards]
        score_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

        returns_values = returns_from_rewards(rewards)
        returns_tensor = torch.tensor(returns_values, dtype=torch.float32).unsqueeze(-1)

        timesteps_tensor = torch.tensor(timesteps, dtype=torch.long)

        return TrajectoryTensors(
            states=states_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            returns_to_go=returns_tensor,
            timesteps=timesteps_tensor,
            hpwl=hpwl_tensor,
            slw=slw_tensor,
            nslw=nslw_tensor,
            score=score_tensor,
            hi=torch.from_numpy(self._compute_hi_features()),
        )


class ExpertTrajectoryDataset(Dataset):
    """PyTorch dataset that streams expert trajectories with caching."""

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        train_ratio: float = 0.95,
        grid_size: Optional[int] = None,  # None = auto-calculate based on board size
        margin: int = 2,
        seed: int = 42,
        cache_dir: Optional[Path] = None,
        preload: bool = False,
        placement_mode: str = "heuristic",
        trajectory_filter: Optional[List[str]] = None,  # List of trajectory names to include (e.g., ['expert1011_traj', ...])
    ) -> None:
        self.root_dir = Path(root_dir)
        self.grid_size = grid_size
        self.margin = margin
        self.split = split
        self.placement_mode = placement_mode
        self.trajectory_filter = trajectory_filter

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        trajectory_dirs = sorted([p for p in self.root_dir.glob("expert*_traj") if p.is_dir()])
        if not trajectory_dirs:
            raise FileNotFoundError(f"No expert trajectories found under {self.root_dir}")

        # Filter trajectories if specified
        if self.trajectory_filter is not None:
            trajectory_names = [p.name for p in trajectory_dirs]
            filtered_dirs = []
            for traj_name in self.trajectory_filter:
                if traj_name in trajectory_names:
                    idx = trajectory_names.index(traj_name)
                    filtered_dirs.append(trajectory_dirs[idx])
            trajectory_dirs = filtered_dirs
            print(f"Filtered to {len(trajectory_dirs)} trajectories from provided filter list")

        if not trajectory_dirs:
            raise ValueError(f"No trajectories remaining after filtering")

        rng = np.random.default_rng(seed)
        rng.shuffle(trajectory_dirs)

        split_idx = int(len(trajectory_dirs) * train_ratio)
        if split == "train":
            selected = trajectory_dirs[:split_idx]
        else:
            selected = trajectory_dirs[split_idx:]

        if not selected:
            raise ValueError(f"Split '{split}' produced no samples; adjust train_ratio")

        self.trajectory_dirs = selected
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._sample_cache: Dict[int, TrajectoryTensors] = {}
        self._processor_cache: Dict[int, ExpertTrajectoryProcessor] = {}

        if preload:
            for idx in range(len(self.trajectory_dirs)):
                self._load_sample(idx)

    def __len__(self) -> int:
        return len(self.trajectory_dirs)

    def __getitem__(self, index: int) -> TrajectoryTensors:
        if not (0 <= index < len(self.trajectory_dirs)):
            raise IndexError(index)
        if index not in self._sample_cache:
            self._load_sample(index)
        return self._sample_cache[index]

    # ------------------------------------------------------------------
    def _cache_path(self, trajectory_dir: Path) -> Optional[Path]:
        if self.cache_dir is None or self.grid_size is None:
            return None  # No caching for dynamic grid sizes
        key = f"{trajectory_dir.name}_gs{self.grid_size}_m{self.margin}.pt"
        return self.cache_dir / key

    def _load_sample(self, index: int) -> None:
        trajectory_dir = self.trajectory_dirs[index]
        cache_path = self._cache_path(trajectory_dir)

        if cache_path is not None and cache_path.exists():
            tensors = torch.load(cache_path, weights_only=False)
            self._sample_cache[index] = tensors
            return

        processor = ExpertTrajectoryProcessor(
            trajectory_dir=trajectory_dir,
            grid_size=self.grid_size,
            margin=self.margin,
            placement_mode=self.placement_mode,
        )
        tensors = processor.rollout()

        if cache_path is not None:
            torch.save(tensors, cache_path)

        self._processor_cache[index] = processor
        self._sample_cache[index] = tensors

    def get_processor(self, index: int) -> ExpertTrajectoryProcessor:
        if not (0 <= index < len(self.trajectory_dirs)):
            raise IndexError(index)
        if index not in self._processor_cache:
            self._load_sample(index)
        return self._processor_cache[index]

    @staticmethod
    def collate_fn(batch: Sequence[TrajectoryTensors]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch passed to collate_fn")

        max_len = max(sample.states.size(0) for sample in batch)
        batch_size = len(batch)
        grid_size = batch[0].states.size(-1)

        states = torch.zeros(batch_size, max_len, 4, grid_size, grid_size, dtype=torch.float32)
        actions = torch.zeros(batch_size, max_len, 3, dtype=torch.long)
        rewards = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        returns = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        hpwls = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        slws = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        nslws = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        scores = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
        timesteps = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        hi = torch.zeros(batch_size, ExpertTrajectoryProcessor.HI_FEATURE_DIM, dtype=torch.float32)

        for b_idx, sample in enumerate(batch):
            length = sample.states.size(0)
            states[b_idx, :length] = sample.states
            actions[b_idx, :length] = sample.actions
            rewards[b_idx, :length] = sample.rewards
            returns[b_idx, :length] = sample.returns_to_go
            hpwls[b_idx, :length] = sample.hpwl
            slws[b_idx, :length] = sample.slw
            nslws[b_idx, :length] = sample.nslw
            scores[b_idx, :length] = sample.score
            timesteps[b_idx, :length] = sample.timesteps
            attention_mask[b_idx, :length] = 1
            hi[b_idx] = sample.hi

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
            "hpwl": hpwls,
            "slw": slws,
            "nslw": nslws,
            "score": scores,
            "hi": hi,
        }
