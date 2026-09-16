#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer with legacy PCB policy checkpoint.

This is for checkpoints whose state_dict keys look like:
  backbone.0.weight
  backbone.0.bias
  backbone.2.weight
  backbone.2.bias
  policy_head.weight
  policy_head.bias
  value_head.weight
  value_head.bias

Usage:
  python infer_legacy_policy.py \
    --checkpoint runs/checkpoints/final_model.pt \
    --input data/benchmark_infer \
    --out runs/infer_final_model_legacy \
    --device cuda \
    --deterministic
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from env import PCBPlacementEnv


class LegacyPolicyNet(nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()

        required = [
            "backbone.0.weight", "backbone.0.bias",
            "backbone.2.weight", "backbone.2.bias",
            "policy_head.weight", "policy_head.bias",
            "value_head.weight", "value_head.bias",
        ]
        missing = [k for k in required if k not in state_dict]
        if missing:
            raise ValueError(
                "This does not look like the legacy policy checkpoint. "
                f"Missing keys: {missing}. First keys: {list(state_dict)[:12]}"
            )

        h1, input_dim = state_dict["backbone.0.weight"].shape
        h2, h1b = state_dict["backbone.2.weight"].shape
        action_dim, hpol = state_dict["policy_head.weight"].shape

        if h1b != h1 or hpol != h2:
            raise ValueError(
                f"Unexpected legacy shapes: input_dim={input_dim}, h1={h1}, h2={h2}, action_dim={action_dim}"
            )

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)

        self.backbone = nn.Sequential(
            nn.Linear(int(input_dim), int(h1)),
            nn.ReLU(),
            nn.Linear(int(h1), int(h2)),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(int(h2), int(action_dim))
        self.value_head = nn.Linear(int(h2), int(state_dict["value_head.weight"].shape[0]))

        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        z = self.backbone(x)
        return self.policy_head(z), self.value_head(z)


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    cleaned = {}
    for k, v in ckpt.items():
        if torch.is_tensor(v):
            nk = k
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
    return cleaned


def infer_grid_size(input_dim: int, action_dim: int, default_grid: int = 128) -> int:
    candidates = []

    for channels in (4, 3, 2, 1):
        root = int(round(math.sqrt(input_dim / channels)))
        if root > 0 and channels * root * root == input_dim:
            candidates.append(root)

    # common action encodings: N*N*4, N*N, 2N+4
    root = int(round(math.sqrt(action_dim / 4)))
    if root > 0 and 4 * root * root == action_dim:
        candidates.append(root)

    root = int(round(math.sqrt(action_dim)))
    if root > 0 and root * root == action_dim:
        candidates.append(root)

    if action_dim > 4 and (action_dim - 4) % 2 == 0:
        candidates.append((action_dim - 4) // 2)

    # Prefer sane PCB grid sizes.
    for g in candidates:
        if g in (32, 64, 96, 128, 160, 192, 256):
            return int(g)
    return int(candidates[0]) if candidates else int(default_grid)


def normalize_raw_task(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "boundary" in raw and "components" in raw:
        comps = []
        for comp in raw.get("components", []):
            name = comp.get("name") or comp.get("ref")
            if not name:
                continue
            size = comp.get("size", [1, 1])
            w = float(comp.get("w", comp.get("width", size[0] if isinstance(size, list) else 1)))
            h = float(comp.get("h", comp.get("height", size[1] if isinstance(size, list) and len(size) > 1 else 1)))
            pads_out = []
            for pad in comp.get("pads", []) or []:
                if isinstance(pad, dict):
                    xy = pad.get("rel_mm") or pad.get("xy") or pad.get("pos") or [0, 0]
                    pads_out.append([float(xy[0]), float(xy[1])])
                else:
                    pads_out.append([float(pad[0]), float(pad[1])])
            comps.append({"name": name, "type": comp.get("type", "default"), "w": w, "h": h, "pads": pads_out})
        return {
            "boundary": raw.get("boundary", []),
            "components": comps,
            "nets": raw.get("nets", []),
        }

    bbox = raw.get("board", {}).get("bbox_mm", [0.0, 0.0, 100.0, 100.0])
    xmin, ymin, xmax, ymax = map(float, bbox)
    boundary = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    comp_by_ref = {c.get("ref"): c for c in raw.get("components", []) if c.get("ref")}
    sequence = raw.get("graph", {}).get("sequence", [])
    ordered_refs = [r for r in sequence if r in comp_by_ref] + [r for r in comp_by_ref if r not in set(sequence)]

    comps = []
    pad_index_by_ref: Dict[str, Dict[str, int]] = {}

    for ref in ordered_refs:
        comp = comp_by_ref[ref]
        size = comp.get("size_mm", [1.0, 1.0])
        w = float(size[0])
        h = float(size[1])
        pad_names = {}
        pads_out = []
        for idx, pad in enumerate(comp.get("pads", []) or []):
            rel = pad.get("rel_mm", [0.0, 0.0])
            # Convert center-relative pad coordinate to local top-left coordinate.
            px = min(max(float(rel[0]) + w / 2.0, 0.0), max(w, 0.0))
            py = min(max(float(rel[1]) + h / 2.0, 0.0), max(h, 0.0))
            name = str(pad.get("name", idx))
            pad_names[name] = idx
            pads_out.append([px, py])
        pad_index_by_ref[ref] = pad_names
        comps.append({"name": ref, "type": comp.get("type", "default"), "w": w, "h": h, "pads": pads_out})

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

    return {"boundary": boundary, "components": comps, "nets": nets_out}


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
        raise ValueError("Invalid board boundary")

    usable = grid_size - 2 * margin
    scale = usable / max(world_w, world_h)
    board_w_grid = world_w * scale
    board_h_grid = world_h * scale
    offset_x = (grid_size - board_w_grid) / 2.0
    offset_y = (grid_size - board_h_grid) / 2.0

    def world_to_grid_xy(x: float, y: float):
        return (x - world_min_x) * scale + offset_x, (y - world_min_y) * scale + offset_y

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

        pads = []
        for pad in comp.get("pads", []) or []:
            px = int(round(float(pad[0]) * scale))
            py = int(round(float(pad[1]) * scale))
            px = min(max(px, 0), max(w_cells - 1, 0))
            py = min(max(py, 0), max(h_cells - 1, 0))
            pads.append([px, py])

        comp_list.append({
            "comp_id": comp_id,
            "name": name,
            "type_id": type_id,
            "size": [w_cells, h_cells],
            "pad_list": pads,
        })
        name_to_id[name] = comp_id

    netlist = []
    for net in task.get("nets", []) or []:
        processed = []
        for entry in net:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            cid = name_to_id.get(entry[0])
            if cid is not None:
                processed.append((cid, int(entry[1])))
        if len(processed) >= 2:
            netlist.append(processed)

    max_pads = max([len(c["pad_list"]) for c in comp_list] + [1])
    num_component_types = max([c["type_id"] for c in comp_list] + [0]) + 1
    mapping = {
        "world_min_x": world_min_x,
        "world_min_y": world_min_y,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }
    return comp_list, boundary_grid, netlist, mapping, max_pads, num_component_types


def grid_to_world_top_left(gx: int, gy: int, mapping: Dict[str, float]):
    wx = mapping["world_min_x"] + (float(gx) - mapping["offset_x"]) / mapping["scale"]
    wy = mapping["world_min_y"] + (float(gy) - mapping["offset_y"]) / mapping["scale"]
    return float(wx), float(wy)


def build_state_vector(obs: Dict[str, Any], input_dim: int) -> np.ndarray:
    view = np.asarray(obs["view_mask"], dtype=np.float32)
    pos = np.asarray(obs["position_mask"], dtype=np.float32)
    wire = np.asarray(obs["wire_mask"], dtype=np.float32)

    # position_mask convention in this env: 0 means legal, 1 means illegal.
    legal_any = (pos == 0).any(axis=0).astype(np.float32)
    illegal_all = 1.0 - legal_any

    if wire.ndim == 3:
        wire_primary = np.min(wire, axis=0).astype(np.float32)
    else:
        wire_primary = wire.astype(np.float32)
    denom = float(np.max(np.abs(wire_primary))) if wire_primary.size else 0.0
    if denom > 1e-6:
        wire_primary = wire_primary / denom
    else:
        wire_primary = np.zeros_like(view, dtype=np.float32)

    # Try common legacy channel formats.
    grid = view.shape[0]
    channels = None
    if input_dim >= 4 * grid * grid:
        # Legacy model likely used 4 image channels plus optional extra scalar/graph features.
        # Use the 4 image channels first, then pad zeros for the unknown extra features.
        bonus = np.zeros_like(view, dtype=np.float32)
        legal_idx = legal_any.astype(bool)
        if legal_idx.any():
            vals = wire_primary[legal_idx]
            thresh = np.quantile(vals, 0.25) if vals.size else 0.0
            bonus[legal_idx & (wire_primary <= thresh)] = 1.0
        channels = [view, illegal_all, wire_primary, bonus]
    elif input_dim == 3 * grid * grid:
        # Common old RL state: view, legal, wire.
        channels = [view, legal_any, wire_primary]
    elif input_dim == 2 * grid * grid:
        channels = [view, legal_any]
    elif input_dim == grid * grid:
        channels = [view]
    else:
        # Best effort: start from the richest state and pad/truncate below.
        bonus = np.zeros_like(view, dtype=np.float32)
        channels = [view, legal_any, illegal_all, wire_primary, bonus]

    vec = np.concatenate([c.reshape(-1) for c in channels]).astype(np.float32)
    if vec.size < input_dim:
        vec = np.pad(vec, (0, input_dim - vec.size), mode="constant")
    elif vec.size > input_dim:
        vec = vec[:input_dim]
    return vec


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


def legal_candidates(obs: Dict[str, Any], env: PCBPlacementEnv, limit: int = 20000):
    pos = np.asarray(obs["position_mask"])
    num_rot, height, width = pos.shape
    out = []
    for r in range(num_rot):
        ys, xs = np.where(pos[r] == 0)
        for x, y in zip(xs, ys):
            a = (int(x), int(y), int(r))
            if valid_by_env(env, a):
                out.append(a)
                if len(out) >= limit:
                    return out
    return out


def decode_action_from_logits(logits: torch.Tensor, obs: Dict[str, Any], env: PCBPlacementEnv, grid_size: int, deterministic: bool = True):
    logits_np = logits.detach().float().cpu().numpy().reshape(-1)
    action_dim = logits_np.shape[0]
    cands = legal_candidates(obs, env)
    if not cands:
        return None

    scores = np.zeros((len(cands),), dtype=np.float32)

    if action_dim == grid_size * grid_size * 4:
        # flat class over (rotation, y, x)
        for i, (x, y, r) in enumerate(cands):
            idx = r * grid_size * grid_size + y * grid_size + x
            scores[i] = logits_np[idx]

    elif action_dim == grid_size * grid_size:
        # flat class over (y, x), rotation selected by legal candidate tie-break.
        for i, (x, y, r) in enumerate(cands):
            idx = y * grid_size + x
            scores[i] = logits_np[idx]

    elif action_dim >= 2 * grid_size + 4:
        # segmented logits: x logits, y logits, rotation logits.
        x_logits = logits_np[:grid_size]
        y_logits = logits_np[grid_size:2 * grid_size]
        r_logits = logits_np[2 * grid_size:2 * grid_size + 4]
        for i, (x, y, r) in enumerate(cands):
            scores[i] = x_logits[x] + y_logits[y] + r_logits[r]

    elif action_dim == 4:
        # rotation-only model, choose best rotation then a low-wire legal position.
        wire = np.asarray(obs["wire_mask"], dtype=np.float32)
        wire_map = np.min(wire, axis=0) if wire.ndim == 3 else wire
        for i, (x, y, r) in enumerate(cands):
            # Smaller wire is better, so subtract it.
            scores[i] = logits_np[r] - 0.01 * wire_map[y, x]

    else:
        # Unknown action encoding: select legal position closest to the model's top class modulo board.
        top = int(np.argmax(logits_np))
        tx = top % grid_size
        ty = (top // grid_size) % grid_size
        tr = (top // max(grid_size * grid_size, 1)) % 4
        for i, (x, y, r) in enumerate(cands):
            scores[i] = -abs(x - tx) - abs(y - ty) - 2 * abs(r - tr)

    if deterministic:
        return cands[int(np.argmax(scores))]

    # Sample from legal candidates only.
    s = scores - np.max(scores)
    p = np.exp(s)
    p = p / max(np.sum(p), 1e-12)
    idx = int(np.random.choice(np.arange(len(cands)), p=p))
    return cands[idx]


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
    ax.set_title("Legacy policy inference placement")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def rollout_one(model, task: Dict[str, Any], out_prefix: Path, grid_size: int, device: torch.device, deterministic: bool, margin: int, render: bool):
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
    placements = {}
    actions = []
    infos = []

    for step_idx in range(len(comp_list)):
        vec = build_state_vector(obs, model.input_dim)
        x = torch.from_numpy(vec).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value = model(x)

        action = decode_action_from_logits(
            logits=logits[0],
            obs=obs,
            env=env,
            grid_size=grid_size,
            deterministic=deterministic,
        )
        if action is None:
            print(f"[WARN] no legal action at step {step_idx}; stop early")
            break

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

        actions.append({
            "step": step_idx,
            "action": [int(action[0]), int(action[1]), int(action[2])],
            "reward": float(reward),
            "value": float(value.detach().cpu().reshape(-1)[0]),
        })
        infos.append(info if isinstance(info, dict) else {})
        obs = obs_next

        if done:
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
        "actions": actions,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if render:
        render_layout(task, placements, out_prefix.with_suffix(".png"))

    return metrics, layout_path


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
    ap.add_argument("--input", default="data/benchmark_infer")
    ap.add_argument("--out", default="runs/infer_final_model_legacy")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--grid-size", type=int, default=None)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch_load(args.checkpoint, device)
    sd = extract_state_dict(ckpt)

    print(f"[INFO] checkpoint={args.checkpoint}")
    print(f"[INFO] first checkpoint keys: {list(sd)[:12]}")

    model = LegacyPolicyNet(sd).to(device)
    model.eval()

    grid_size = int(args.grid_size or infer_grid_size(model.input_dim, model.action_dim, 128))
    print(f"[INFO] legacy input_dim={model.input_dim}, action_dim={model.action_dim}, grid_size={grid_size}, device={device}")

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
            metrics, layout_path = rollout_one(
                model=model,
                task=task,
                out_prefix=out_dir / f.stem,
                grid_size=grid_size,
                device=device,
                deterministic=args.deterministic,
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
