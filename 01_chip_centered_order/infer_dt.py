#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer PCB placement with a trained DecisionTransformer checkpoint.

Typical usage:
  python infer_dt.py \
    --checkpoint runs/checkpoints/final_model.pt \
    --input data/benchmark_infer \
    --out runs/infer_final_model \
    --device cuda \
    --deterministic
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env import PCBPlacementEnv
from DT.models.dt_model import DecisionTransformer


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def infer_model_config(state_dict: Dict[str, torch.Tensor], ckpt: Any, args) -> Tuple[int, int, int]:
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    hidden_size = args.hidden_size or cfg.get("hidden_size")
    if hidden_size is None:
        if "transformer.wte.weight" in state_dict:
            hidden_size = int(state_dict["transformer.wte.weight"].shape[1])
        elif "embed_state.conv.0.weight" in state_dict:
            hidden_size = int(state_dict["embed_state.conv.0.weight"].shape[0])
        else:
            hidden_size = 256

    grid_size = args.grid_size or cfg.get("grid_size")
    if grid_size is None:
        if "embed_x.weight" in state_dict:
            grid_size = int(state_dict["embed_x.weight"].shape[0])
        elif "predict_x.bias" in state_dict:
            grid_size = int(state_dict["predict_x.bias"].shape[0])
        else:
            grid_size = 128

    hi_dim = args.hi_dim or cfg.get("hi_dim")
    if hi_dim is None:
        if "embed_hi.weight" in state_dict:
            hi_dim = int(state_dict["embed_hi.weight"].shape[1])
        else:
            hi_dim = 11

    return int(grid_size), int(hidden_size), int(hi_dim)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_raw_task(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert either:
      1) old task format: {boundary, components: [{name,w,h,pads}], nets: [[[name,pad_idx], ...]]}
      2) benchmark_infer format: {board:{bbox_mm}, components:[{ref,size_mm,pads}], nets:{net:[ref.pad]}}
    into a common world-coordinate task format.
    """
    if "boundary" in raw and "components" in raw:
        comps = []
        for comp in raw.get("components", []):
            name = comp.get("name") or comp.get("ref")
            if not name:
                continue
            w = float(comp.get("w", comp.get("width", comp.get("size", [1, 1])[0] if isinstance(comp.get("size"), list) else 1)))
            h = float(comp.get("h", comp.get("height", comp.get("size", [1, 1])[1] if isinstance(comp.get("size"), list) and len(comp.get("size")) > 1 else 1)))
            pads = comp.get("pads", []) or comp.get("pad_list", [])
            pad_xy = []
            pad_names = {}
            for i, pad in enumerate(pads):
                if isinstance(pad, dict):
                    xy = pad.get("rel_mm") or pad.get("xy") or pad.get("pos") or [0, 0]
                    pad_name = str(pad.get("name", i))
                    px, py = float(xy[0]), float(xy[1])
                else:
                    pad_name = str(i)
                    px, py = float(pad[0]), float(pad[1])
                pad_names[pad_name] = i
                pad_xy.append([px, py])
            comps.append({"name": name, "type": comp.get("type", "default"), "w": w, "h": h, "pads": pad_xy, "_pad_names": pad_names})
        return {
            "boundary": raw.get("boundary", []),
            "components": comps,
            "nets": raw.get("nets", []),
            "min_spacing": raw.get("min_spacing", 0.0),
            "min_pad_spacing": raw.get("min_pad_spacing", 0.0),
        }

    if "board" not in raw or "components" not in raw:
        raise ValueError("Unsupported input JSON format: missing boundary/components or board/components")

    bbox = raw.get("board", {}).get("bbox_mm", [0.0, 0.0, 100.0, 100.0])
    xmin, ymin, xmax, ymax = map(float, bbox)
    boundary = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    comp_by_ref = {c.get("ref"): c for c in raw.get("components", []) if c.get("ref")}
    sequence = raw.get("graph", {}).get("sequence", [])
    ordered_refs = [ref for ref in sequence if ref in comp_by_ref]
    ordered_refs += [ref for ref in comp_by_ref if ref not in set(ordered_refs)]

    comps = []
    pad_index_by_ref: Dict[str, Dict[str, int]] = {}

    for ref in ordered_refs:
        comp = comp_by_ref[ref]
        size = comp.get("size_mm", [1.0, 1.0])
        w = float(size[0])
        h = float(size[1])
        pads_out = []
        pad_names = {}

        for idx, pad in enumerate(comp.get("pads", []) or []):
            rel = pad.get("rel_mm", [0.0, 0.0])
            # benchmark pads are usually relative to component center; env expects local bbox coords.
            px = clamp(float(rel[0]) + w / 2.0, 0.0, max(w, 0.0))
            py = clamp(float(rel[1]) + h / 2.0, 0.0, max(h, 0.0))
            name = str(pad.get("name", idx))
            pad_names[name] = idx
            pads_out.append([px, py])

        pad_index_by_ref[ref] = pad_names
        comps.append({
            "name": ref,
            "type": comp.get("type", "default"),
            "w": w,
            "h": h,
            "pads": pads_out,
            "_pad_names": pad_names,
        })

    nets_out = []
    for _, pins in (raw.get("nets", {}) or {}).items():
        net = []
        for pin in pins:
            if not isinstance(pin, str) or "." not in pin:
                continue
            ref, pad_name = pin.rsplit(".", 1)
            if ref not in pad_index_by_ref:
                continue
            pidx = pad_index_by_ref[ref].get(str(pad_name))
            if pidx is None:
                try:
                    pidx = int(pad_name) - 1
                except Exception:
                    pidx = None
            if pidx is not None and pidx >= 0:
                net.append([ref, int(pidx)])
        if len(net) >= 2:
            nets_out.append(net)

    return {
        "boundary": boundary,
        "components": comps,
        "nets": nets_out,
        "min_spacing": 0.0,
        "min_pad_spacing": 0.0,
    }


def convert_task_to_env_struct(task: Dict[str, Any], grid_size: int, margin: int = 2):
    xs = [float(p[0]) for p in task.get("boundary", [])]
    ys = [float(p[1]) for p in task.get("boundary", [])]
    if not xs or not ys:
        raise ValueError("Task boundary is empty")

    world_min_x = min(xs)
    world_min_y = min(ys)
    world_w = max(xs) - world_min_x
    world_h = max(ys) - world_min_y
    if world_w <= 0 or world_h <= 0:
        raise ValueError("Invalid board boundary dimensions")

    usable = grid_size - 2 * margin
    if usable <= 0:
        raise ValueError("grid_size too small for margin")

    scale = usable / max(world_w, world_h)
    board_w_grid = world_w * scale
    board_h_grid = world_h * scale
    offset_x = (grid_size - board_w_grid) / 2.0
    offset_y = (grid_size - board_h_grid) / 2.0

    def world_to_grid_xy(x: float, y: float):
        gx = (x - world_min_x) * scale + offset_x
        gy = (y - world_min_y) * scale + offset_y
        return gx, gy

    boundary_grid = [tuple(world_to_grid_xy(float(x), float(y))) for x, y in task.get("boundary", [])]

    type_to_id: Dict[str, int] = {}
    comp_list = []
    name_to_id = {}

    for comp_id, comp in enumerate(task.get("components", [])):
        name = comp.get("name", f"C{comp_id}")
        type_name = comp.get("type", "default")
        type_id = type_to_id.setdefault(type_name, len(type_to_id))

        w_world = float(comp.get("w", 1.0))
        h_world = float(comp.get("h", 1.0))
        w_cells = max(1, int(round(w_world * scale)))
        h_cells = max(1, int(round(h_world * scale)))

        pad_list = []
        for pad in comp.get("pads", []) or []:
            px = int(round(float(pad[0]) * scale))
            py = int(round(float(pad[1]) * scale))
            px = min(max(px, 0), max(w_cells - 1, 0))
            py = min(max(py, 0), max(h_cells - 1, 0))
            pad_list.append([px, py])

        comp_list.append({
            "comp_id": comp_id,
            "name": name,
            "type_id": type_id,
            "size": [w_cells, h_cells],
            "pad_list": pad_list,
        })
        name_to_id[name] = comp_id

    netlist_indexed = []
    for net in task.get("nets", []) or []:
        processed = []
        for entry in net:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            comp_name, pad_idx = entry
            cid = name_to_id.get(comp_name)
            if cid is None:
                continue
            processed.append((cid, int(pad_idx)))
        if len(processed) >= 2:
            netlist_indexed.append(processed)

    max_pads = max([len(c["pad_list"]) for c in comp_list] + [1])
    num_component_types = max([c["type_id"] for c in comp_list] + [0]) + 1

    mapping = {
        "world_min_x": world_min_x,
        "world_min_y": world_min_y,
        "world_w": world_w,
        "world_h": world_h,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }

    return comp_list, boundary_grid, netlist_indexed, mapping, max_pads, num_component_types


def grid_to_world_top_left(gx: int, gy: int, mapping: Dict[str, float]):
    wx = mapping["world_min_x"] + (float(gx) - mapping["offset_x"]) / mapping["scale"]
    wy = mapping["world_min_y"] + (float(gy) - mapping["offset_y"]) / mapping["scale"]
    return float(wx), float(wy)


def build_state_tensor(obs: Dict[str, Any], bonus_quantile: float = 0.25) -> torch.FloatTensor:
    view_mask = np.asarray(obs["view_mask"]).astype(np.float32)
    position_mask = np.asarray(obs["position_mask"]).astype(np.float32)
    wire_mask = np.asarray(obs["wire_mask"]).astype(np.float32)

    legal_map = (position_mask == 0).any(axis=0)
    wire_map = np.min(wire_mask, axis=0).astype(np.float32)

    # Channel 1: 0=legal, 1=illegal, matching training.
    position_channel = np.where(legal_map, 0.0, 1.0).astype(np.float32)

    denom = float(np.max(np.abs(wire_map))) if wire_map.size else 0.0
    if denom > 1e-6:
        wire_channel = wire_map / denom
    else:
        wire_channel = np.zeros_like(wire_map, dtype=np.float32)

    bonus_mask = np.zeros_like(wire_channel, dtype=np.float32)
    legal_indices = legal_map & np.isfinite(wire_map)
    if legal_indices.any():
        vals = wire_map[legal_indices]
        threshold = np.quantile(vals, bonus_quantile) if vals.size else 0.0
        bonus_mask[legal_indices & (wire_map <= threshold)] = 1.0
    bonus_mask = bonus_mask * (1.0 - position_channel)

    return torch.from_numpy(np.stack([view_mask, position_channel, wire_channel, bonus_mask], axis=0).copy())


def compute_hi_features(comp_list, netlist, boundary_grid, grid_size: int, hi_dim: int) -> torch.FloatTensor:
    comps = comp_list
    comp_count = len(comps)
    widths = np.array([c["size"][0] for c in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)
    heights = np.array([c["size"][1] for c in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)
    areas = widths * heights
    pad_counts = np.array([len(c["pad_list"]) for c in comps], dtype=np.float32) if comp_count else np.array([0.0], dtype=np.float32)

    total_area = float(np.sum(areas))
    grid_area = float(grid_size ** 2)
    avg_area = float(np.mean(areas)) if comp_count else 0.0
    std_area = float(np.std(areas)) if comp_count else 0.0
    max_area = float(np.max(areas)) if comp_count else 0.0

    avg_pad = float(np.mean(pad_counts)) if comp_count else 0.0
    std_pad = float(np.std(pad_counts)) if comp_count else 0.0
    max_pads = max(float(np.max(pad_counts)) if pad_counts.size else 1.0, 1.0)

    net_lengths = [len(n) for n in netlist]
    net_count = len(net_lengths)
    avg_net_degree = float(np.mean(net_lengths)) if net_lengths else 0.0

    xs = [p[0] for p in boundary_grid]
    ys = [p[1] for p in boundary_grid]
    board_w = float(max(max(xs) - min(xs), 1.0))
    board_h = float(max(max(ys) - min(ys), 1.0))
    aspect_ratio = board_w / board_h if board_h > 0 else 1.0
    density = total_area / grid_area if grid_area > 0 else 0.0

    features = np.array([
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
        grid_size / 256.0,
    ], dtype=np.float32)

    if hi_dim != len(features):
        padded = np.zeros((hi_dim,), dtype=np.float32)
        n = min(hi_dim, len(features))
        padded[:n] = features[:n]
        features = padded

    return torch.from_numpy(features)


def valid_by_env(env: PCBPlacementEnv, action: Tuple[int, int, int]) -> bool:
    x, y, rot = map(int, action)
    if env.current_component is None:
        return False
    try:
        w, h, _ = env._get_rotated_dims_and_pads(env.current_component, rot)
        return (
            env._check_boundary(x, y, w, h)
            and env._check_overlap(x, y, w, h, rot)
            and env._check_spacing(x, y, w, h, rot)
        )
    except Exception:
        return False


def ensure_legal_action(env: PCBPlacementEnv, obs: Dict[str, Any], action: Tuple[int, int, int], max_candidates: int = 20000):
    x, y, rot = map(int, action)
    position_mask = np.asarray(obs.get("position_mask"))
    num_rot, height, width = position_mask.shape

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    rot = max(0, min(num_rot - 1, rot))
    candidate = (x, y, rot)

    if position_mask[rot, y, x] == 0 and valid_by_env(env, candidate):
        return candidate

    candidates = []
    for r in range(num_rot):
        ys, xs = np.where(position_mask[r] == 0)
        if len(xs) == 0:
            continue
        dist = np.abs(xs - x) + np.abs(ys - y) + (r != rot) * 2
        order = np.argsort(dist)
        for idx in order[:max_candidates]:
            candidates.append((int(dist[idx]), int(xs[idx]), int(ys[idx]), int(r)))

    candidates.sort(key=lambda t: t[0])
    for _, cx, cy, cr in candidates[:max_candidates]:
        cand = (cx, cy, cr)
        if valid_by_env(env, cand):
            return cand

    return None


def rollout_one(model, task: Dict[str, Any], out_prefix: Path, grid_size: int, hi_dim: int, device: torch.device,
                deterministic: bool, temperature: float, margin: int, render: bool):
    comp_list, boundary_grid, netlist, mapping, max_pads, num_component_types = convert_task_to_env_struct(task, grid_size, margin)

    env = PCBPlacementEnv(
        grid_size=grid_size,
        boundary_polygon=boundary_grid,
        component_list=comp_list,
        netlist=netlist,
        max_pads=max_pads,
        num_component_types=num_component_types,
    )

    obs = env.reset()
    hi_features = compute_hi_features(comp_list, netlist, boundary_grid, grid_size, hi_dim).to(device)

    states_history = []
    actions_history = []
    returns_history = []
    timesteps_history = []

    placements = {}
    actions_taken = []
    infos = []

    step_idx = 0
    rtg_remaining = 0.0

    while True:
        state_tensor = build_state_tensor(obs).to(dtype=torch.float32)
        states_history.append(state_tensor)
        actions_history.append(torch.zeros(3, dtype=torch.long))
        returns_history.append(torch.tensor([rtg_remaining], dtype=torch.float32))
        timesteps_history.append(step_idx)

        states_seq = torch.stack(states_history, dim=0).to(device)
        actions_seq = torch.stack(actions_history, dim=0).to(device)
        returns_seq = torch.stack(returns_history, dim=0).to(device)
        timesteps_seq = torch.tensor(timesteps_history, dtype=torch.long, device=device)

        with torch.no_grad():
            raw_action = model.get_action(
                states=states_seq,
                actions=actions_seq,
                rewards=None,
                returns_to_go=returns_seq,
                timesteps=timesteps_seq,
                deterministic=deterministic,
                temperature=temperature,
                hi=hi_features.unsqueeze(0),
            )

        action = ensure_legal_action(env, obs, raw_action)
        if action is None:
            print(f"[WARN] no legal action at step {step_idx}; stop early")
            break

        actions_history[-1] = torch.tensor(action, dtype=torch.long)

        step_result = env.step(action)
        if len(step_result) == 5:
            obs_next, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            obs_next, reward, done, info = step_result

        if isinstance(info, dict) and info.get("illegal"):
            print(f"[WARN] env returned illegal at step {step_idx}: {info}; stop early")
            break

        if env.placed_components:
            px, py, pw, ph, type_id, comp_id = env.placed_components[-1]
            name = comp_list[int(comp_id)]["name"]
            wx, wy = grid_to_world_top_left(int(px), int(py), mapping)
            placements[name] = {
                "x": wx,
                "y": wy,
                "rot": int(action[2]),
                "side": "top",
            }

        actions_taken.append({
            "step": step_idx,
            "raw_action": [int(raw_action[0]), int(raw_action[1]), int(raw_action[2])],
            "action": [int(action[0]), int(action[1]), int(action[2])],
            "reward": float(reward),
        })
        infos.append(info if isinstance(info, dict) else {})

        obs = obs_next
        step_idx += 1

        if done or step_idx >= len(comp_list):
            break

    layout_path = out_prefix.with_suffix(".layout.json")
    metrics_path = out_prefix.with_suffix(".metrics.json")
    with layout_path.open("w", encoding="utf-8") as f:
        json.dump(placements, f, ensure_ascii=False, indent=2)

    final_info = infos[-1] if infos else {}
    metrics = {
        "num_components": len(comp_list),
        "num_placed": len(placements),
        "complete": len(placements) == len(comp_list),
        "grid_size": grid_size,
        "hpwl": final_info.get("hpwl"),
        "slw": final_info.get("slw"),
        "nslw": final_info.get("nslw"),
        "score": final_info.get("score"),
        "actions": actions_taken,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if render:
        render_layout(task, placements, out_prefix.with_suffix(".png"))

    return metrics, layout_path, metrics_path


def render_layout(task: Dict[str, Any], placements: Dict[str, Any], out_png: Path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle
    except Exception as exc:
        print(f"[WARN] skip render: {exc}")
        return

    comps = {c["name"]: c for c in task.get("components", [])}
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)

    boundary = task.get("boundary", [])
    if boundary:
        ax.add_patch(Polygon(boundary, closed=True, facecolor="#1b5e20", edgecolor="#0b3d0b", alpha=0.85))

    for name, p in placements.items():
        comp = comps.get(name)
        if not comp:
            continue
        x = float(p["x"])
        y = float(p["y"])
        w = float(comp.get("w", 1.0))
        h = float(comp.get("h", 1.0))
        rot_deg = int(p.get("rot", 0)) * 90
        ax.add_patch(Rectangle((x, y), w, h, angle=rot_deg, facecolor="tab:blue", edgecolor="black", alpha=0.65))
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=5)

    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("DT inference placement")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def iter_input_files(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.json"))
    raise FileNotFoundError(input_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/checkpoints/final_model.pt")
    ap.add_argument("--input", default="data/benchmark_infer", help="A JSON file or a directory of JSON files")
    ap.add_argument("--out", default="runs/infer_final_model")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--grid-size", type=int, default=None, help="Override grid size; default inferred from checkpoint")
    ap.add_argument("--hidden-size", type=int, default=None, help="Override hidden size; default inferred from checkpoint")
    ap.add_argument("--hi-dim", type=int, default=None, help="Override hi feature dim; default inferred from checkpoint")
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch_load(args.checkpoint, device)
    state_dict = extract_state_dict(ckpt)
    grid_size, hidden_size, hi_dim = infer_model_config(state_dict, ckpt, args)

    print(f"[INFO] checkpoint={args.checkpoint}")
    print(f"[INFO] grid_size={grid_size}, hidden_size={hidden_size}, hi_dim={hi_dim}, device={device}")

    model = DecisionTransformer(grid_size=grid_size, hidden_size=hidden_size, hi_dim=hi_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}; examples={missing[:8]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}; examples={unexpected[:8]}")

    model.to(device)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(args.input)
    if not files:
        raise RuntimeError(f"No json files found under {args.input}")

    summary = []
    for f in files:
        try:
            raw = json.load(open(f, "r", encoding="utf-8"))
            task = normalize_raw_task(raw)
            metrics, layout_path, metrics_path = rollout_one(
                model=model,
                task=task,
                out_prefix=out_dir / f.stem,
                grid_size=grid_size,
                hi_dim=hi_dim,
                device=device,
                deterministic=args.deterministic,
                temperature=args.temperature,
                margin=args.margin,
                render=not args.no_render,
            )
            print(f"[OK] {f.name}: placed {metrics['num_placed']}/{metrics['num_components']} -> {layout_path}")
            summary.append({"input": str(f), **{k: metrics.get(k) for k in ("num_components", "num_placed", "complete", "hpwl", "slw", "score")}})
        except Exception as exc:
            print(f"[FAIL] {f}: {exc}")
            summary.append({"input": str(f), "error": str(exc)})

    with (out_dir / "summary.json").open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"[DONE] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
