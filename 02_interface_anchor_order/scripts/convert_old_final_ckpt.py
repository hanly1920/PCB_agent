from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from pcbplace.model import ModelConfig
from pcbplace.region_prior import (
    REGION_TYPE_NAMES,
    SEMANTIC_CLASS_NAMES,
    SIDE_PREFERENCE_NAMES,
    SUBZONE_NAMES,
    PAIRWISE_RELATION_NAMES,
)


PREFIXES = (
    "model_state.",
    "state_dict.",
    "module.",
    "_orig_mod.",
    "model.",
    "policy.",
)


def is_tensor(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def flatten_tensors(obj: Any, prefix: str = "", out: Dict[str, Any] | None = None):
    if out is None:
        out = {}

    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        obj = obj.state_dict()

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_tensors(v, key, out)
    elif is_tensor(obj):
        out[prefix] = obj

    return out


def strip_prefixes(key: str) -> str:
    k = str(key)
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def normalize_state(flat: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in flat.items():
        nk = strip_prefixes(k)
        out[nk] = v
    return out


def find_payload_candidates(obj: Any, path: str = "ckpt", out=None):
    if out is None:
        out = []

    if isinstance(obj, dict):
        score = 0
        if "model_state" in obj:
            score += 100
        if "state_dict" in obj:
            score += 90
        if "obs_dim" in obj:
            score += 30
        if "model_cfg" in obj:
            score += 20

        tensor_count = sum(1 for _k, v in obj.items() if is_tensor(v))
        score += tensor_count

        if score > 0:
            out.append((score, path, obj))

        for k, v in obj.items():
            if isinstance(v, dict) or hasattr(v, "state_dict"):
                find_payload_candidates(v, f"{path}.{k}", out)

    elif hasattr(obj, "state_dict"):
        out.append((80, path, obj))

    return out


def pick_payload(raw: Any) -> Tuple[str, Any]:
    cands = find_payload_candidates(raw)
    if not cands:
        return "ckpt", raw
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1], cands[0][2]


def get_meta(raw: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    meta = {}

    if isinstance(raw, dict):
        if isinstance(raw.get("meta"), dict):
            meta.update(raw["meta"])
        for k, v in raw.items():
            if k not in {"model", "model_state", "state_dict", "optimizer"}:
                if not isinstance(v, dict) and not is_tensor(v):
                    meta.setdefault(k, v)

    if isinstance(payload, dict):
        if isinstance(payload.get("meta"), dict):
            meta.update(payload["meta"])
        for k, v in payload.items():
            if k not in {"model", "model_state", "state_dict", "optimizer"}:
                if not isinstance(v, dict) and not is_tensor(v):
                    meta.setdefault(k, v)

    return meta


def extract_state(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if "model_state" in payload:
            obj = payload["model_state"]
        elif "state_dict" in payload:
            obj = payload["state_dict"]
        elif "model" in payload:
            obj = payload["model"]
        else:
            obj = payload
    else:
        obj = payload

    flat = flatten_tensors(obj)
    return normalize_state(flat)


def tensor_out_dim(state: Dict[str, Any], key: str, default: int) -> int:
    t = state.get(key)
    if t is None or not is_tensor(t) or len(t.shape) < 1:
        return int(default)
    return int(t.shape[0])


def infer_num_layers(state: Dict[str, Any]) -> int:
    layers = []
    for k in state.keys():
        parts = str(k).split(".")
        if len(parts) >= 3 and parts[0] == "tr" and parts[1] == "layers":
            try:
                layers.append(int(parts[2]))
            except ValueError:
                pass
    return max(layers) + 1 if layers else 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    raw = torch.load(args.src, map_location="cpu")

    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint must be dict, got {type(raw)}")

    payload_path, payload = pick_payload(raw)
    state = extract_state(payload)
    meta = get_meta(raw, payload)

    print(f"[convert] picked payload: {payload_path}")
    print(f"[convert] tensor keys found: {len(state)}")

    obs_w = state.get("obs_proj.weight")

    if obs_w is None:
        print("\n[convert] Cannot find obs_proj.weight.")
        print("[convert] This means final_model.pt is probably not the same MaskedPolicy architecture.")
        print("\n[convert] First 120 tensor keys:")
        for i, (k, v) in enumerate(state.items()):
            if i >= 120:
                break
            shape = tuple(v.shape) if is_tensor(v) else None
            print(f"  {k}: {shape}")

        raise SystemExit(
            "\nStop here. Send the tensor-key output above, or use the old pcb_autoplace "
            "project's own infer script for this final_model.pt."
        )

    d_model = int(obs_w.shape[0])
    obs_dim = int(meta.get("obs_dim", obs_w.shape[1]))

    action_w = state.get("k_mlp.0.weight")
    action_feat_dim = int(action_w.shape[1]) if action_w is not None else int(meta.get("action_feat_dim", 4))

    cfg_src = {}
    if isinstance(meta.get("model_cfg"), dict):
        cfg_src = dict(meta["model_cfg"])

    cfg_src["d_model"] = int(cfg_src.get("d_model", d_model))
    cfg_src["num_layers"] = int(cfg_src.get("num_layers", infer_num_layers(state)))
    cfg_src["dropout"] = float(cfg_src.get("dropout", 0.1))

    nhead = int(cfg_src.get("nhead", 8))
    if d_model % nhead != 0:
        for cand in [8, 4, 2, 1]:
            if d_model % cand == 0:
                nhead = cand
                break
    cfg_src["nhead"] = nhead

    # validate cfg fields only
    allowed = set(ModelConfig.__dataclass_fields__.keys())
    model_cfg = {k: v for k, v in cfg_src.items() if k in allowed}

    out = dict(meta)
    out.update({
        "model_state": state,
        "obs_dim": obs_dim,
        "model_cfg": model_cfg,
        "action_feat_dim": action_feat_dim,
        "region_grid_shape": meta.get("region_grid_shape", [6, 6]),
        "num_region_types": tensor_out_dim(
            state,
            "region_type_head.2.weight",
            len(REGION_TYPE_NAMES),
        ),
        "num_semantic_classes": tensor_out_dim(
            state,
            "semantic_class_head.2.weight",
            len(SEMANTIC_CLASS_NAMES),
        ),
        "num_side_preferences": tensor_out_dim(
            state,
            "side_preference_head.2.weight",
            len(SIDE_PREFERENCE_NAMES),
        ),
        "num_subzones": tensor_out_dim(
            state,
            "subzone_head.2.weight",
            len(SUBZONE_NAMES),
        ),
        "num_pairwise_relations": tensor_out_dim(
            state,
            "pairwise_relation_mlp.2.weight",
            len(PAIRWISE_RELATION_NAMES),
        ),
    })

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    print(f"[convert] saved: {dst}")
    print(f"[convert] obs_dim={obs_dim}")
    print(f"[convert] d_model={model_cfg['d_model']}")
    print(f"[convert] nhead={model_cfg['nhead']}")
    print(f"[convert] num_layers={model_cfg['num_layers']}")
    print(f"[convert] action_feat_dim={action_feat_dim}")


if __name__ == "__main__":
    main()
