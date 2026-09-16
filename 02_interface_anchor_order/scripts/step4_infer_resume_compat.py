from __future__ import annotations

import argparse
import glob
import traceback
from pathlib import Path
from typing import Any, Dict

import torch

import pcbplace.infer as infer_mod
from pcbplace.model import MaskedPolicy, ModelConfig
from pcbplace.region_prior import (
    REGION_TYPE_NAMES,
    SEMANTIC_CLASS_NAMES,
    SIDE_PREFERENCE_NAMES,
    SUBZONE_NAMES,
    PAIRWISE_RELATION_NAMES,
)
from pcbplace.utils import save_json


def _normalize_state_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    for key, value in state.items():
        k = str(key)

        changed = True
        while changed:
            changed = False
            for prefix in [
                "module.",
                "model.",
                "policy.",
                "net.",
                "_orig_mod.",
            ]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True

        out[k] = value

    return out


def _extract_state_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state" in raw:
        state = raw["model_state"]
    elif "model" in raw:
        state = raw["model"]
    elif "state_dict" in raw:
        state = raw["state_dict"]
    else:
        raise KeyError(
            "Cannot find model weights. Expected one of: "
            "model_state, model, state_dict"
        )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"model state must be dict, got {type(state)}")

    return _normalize_state_keys(state)


def _get_tensor(state: Dict[str, Any], name: str):
    if name in state:
        return state[name]

    matches = [
        k for k in state.keys()
        if str(k).endswith("." + name) or str(k).endswith(name)
    ]

    if matches:
        return state[sorted(matches, key=len)[0]]

    return None


def _out_dim(state: Dict[str, Any], key: str, default: int) -> int:
    t = _get_tensor(state, key)
    if t is None or not hasattr(t, "shape") or len(t.shape) < 1:
        return int(default)
    return int(t.shape[0])


def _infer_num_layers(state: Dict[str, Any]) -> int:
    layers = []

    for key in state.keys():
        parts = str(key).split(".")
        if len(parts) >= 3 and parts[0] == "tr" and parts[1] == "layers":
            try:
                layers.append(int(parts[2]))
            except ValueError:
                pass

    if not layers:
        return 6

    return max(layers) + 1


def _compat_load_model(ckpt_path: str, device: str = "cpu"):
    raw = torch.load(ckpt_path, map_location=device)

    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint must be dict, got {type(raw)}")

    state = _extract_state_dict(raw)

    obs_w = _get_tensor(state, "obs_proj.weight")
    if obs_w is None:
        print("Available state_dict keys sample:")
        for i, k in enumerate(state.keys()):
            if i >= 50:
                break
            print(" ", k)
        raise KeyError("Cannot infer obs_dim: obs_proj.weight not found")

    d_model = int(obs_w.shape[0])
    obs_dim = int(obs_w.shape[1])

    action_w = _get_tensor(state, "k_mlp.0.weight")
    action_feat_dim = 4
    if action_w is not None and hasattr(action_w, "shape") and len(action_w.shape) >= 2:
        action_feat_dim = int(action_w.shape[1])

    meta = {}
    if isinstance(raw.get("meta"), dict):
        meta.update(raw["meta"])

    for k, v in raw.items():
        if k not in {
            "model",
            "model_state",
            "state_dict",
            "optimizer",
            "optimizer_state",
        }:
            meta.setdefault(k, v)

    cfg_src = {}
    if isinstance(meta.get("model_cfg"), dict):
        cfg_src = meta["model_cfg"]
    elif isinstance(raw.get("model_cfg"), dict):
        cfg_src = raw["model_cfg"]

    allowed = set(ModelConfig.__dataclass_fields__.keys())
    cfg_kwargs = {
        k: v for k, v in cfg_src.items()
        if k in allowed
    }

    cfg = ModelConfig(**cfg_kwargs)
    cfg.d_model = d_model
    cfg.num_layers = _infer_num_layers(state)

    if d_model % int(cfg.nhead) != 0:
        for candidate in [8, 4, 2, 1]:
            if d_model % candidate == 0:
                cfg.nhead = candidate
                break

    num_region_types = _out_dim(
        state,
        "region_type_head.2.weight",
        len(REGION_TYPE_NAMES),
    )
    num_semantic_classes = _out_dim(
        state,
        "semantic_class_head.2.weight",
        len(SEMANTIC_CLASS_NAMES),
    )
    num_side_preferences = _out_dim(
        state,
        "side_preference_head.2.weight",
        len(SIDE_PREFERENCE_NAMES),
    )
    num_subzones = _out_dim(
        state,
        "subzone_head.2.weight",
        len(SUBZONE_NAMES),
    )
    num_pairwise_relations = _out_dim(
        state,
        "pairwise_relation_mlp.2.weight",
        len(PAIRWISE_RELATION_NAMES),
    )

    region_grid_shape = meta.get("region_grid_shape", [6, 6])

    model = MaskedPolicy(
        obs_dim=obs_dim,
        cfg=cfg,
        action_feat_dim=action_feat_dim,
        region_grid_shape=(
            int(region_grid_shape[0]),
            int(region_grid_shape[1]),
        ),
        num_region_types=num_region_types,
        num_semantic_classes=num_semantic_classes,
        num_side_preferences=num_side_preferences,
        num_subzones=num_subzones,
        num_pairwise_relations=num_pairwise_relations,
    )

    current = model.state_dict()
    compatible = {}
    skipped = []

    for k, v in state.items():
        if k in current and tuple(v.shape) == tuple(current[k].shape):
            compatible[k] = v
        else:
            skipped.append(k)

    incompatible = model.load_state_dict(compatible, strict=False)

    if skipped:
        print(
            "[compat] skipped incompatible/unexpected keys:",
            skipped[:20],
            "..." if len(skipped) > 20 else "",
        )

    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
        print(
            "[compat] missing_keys=",
            list(incompatible.missing_keys)[:20],
            "unexpected_keys=",
            list(incompatible.unexpected_keys)[:20],
        )

    model.to(device)
    model.eval()

    meta["model_state"] = compatible
    meta["obs_dim"] = obs_dim
    meta["model_cfg"] = cfg.__dict__
    meta["action_feat_dim"] = action_feat_dim
    meta["num_region_types"] = num_region_types
    meta["num_semantic_classes"] = num_semantic_classes
    meta["num_side_preferences"] = num_side_preferences
    meta["num_subzones"] = num_subzones
    meta["num_pairwise_relations"] = num_pairwise_relations

    teacher = infer_mod._teacher_from_ckpt(meta)
    region_cfg = infer_mod._region_cfg_from_ckpt(meta)

    return model, meta, teacher, region_cfg


infer_mod.load_model = _compat_load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_glob", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(args.test_glob))

    completed = 0
    skipped = 0
    failed = []

    for index, path in enumerate(paths, start=1):
        src = Path(path)
        out_path = out_dir / f"{src.stem}.infer.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{len(paths)}] skip existing: {out_path}")
            skipped += 1
            continue

        try:
            result = infer_mod.infer_layout(
                path,
                args.ckpt,
                device=args.device,
            )

            save_json(result, out_path)

            print(
                f"[{index}/{len(paths)}] "
                f"{path} -> {out_path} "
                f"obj={result['objective']:.3f}"
            )
            completed += 1

        except Exception as exc:
            failed.append((path, repr(exc)))
            print(f"[{index}/{len(paths)}] FAILED: {path}: {exc}")
            traceback.print_exc()

    print()
    print(
        f"total={len(paths)}, "
        f"completed={completed}, "
        f"skipped={skipped}, "
        f"failed={len(failed)}"
    )

    if failed:
        print("\nFailed files:")
        for path, error in failed:
            print(f"  {path}: {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
