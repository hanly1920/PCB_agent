from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pcbplace.infer import infer_layout, LAYOUT_OBJECTIVE_PRESETS
from pcbplace.dataset import SEQUENCE_POLICIES
from pcbplace.utils import save_json

_METRIC_PRINT_KEYS = [
    "hpwl_ratio",
    "min_gap_p10",
    "min_gap_p25",
    "local_density_p90",
    "corner_mass_ratio",
    "edge_non_interface_count",
    "module_centroid_error",
    "module_bbox_iou_with_original",
    "anchor_distance_error",
    "large_component_drift_mm",
    "large_component_drift_ratio",
    "connector_side_accuracy",
    "same_side_order_accuracy",
    "orientation_consistency",
    "pitch_cv",
    "boundary_pitch_error",
    "module_overlap_ratio",
    "decoupling_distance_p90",
    "clock_load_distance_p90",
    "esd_to_connector_distance_p90",
    "power_loop_distance_p90",
]

def _fmt_metrics(metrics: dict) -> str:
    if metrics.get("metrics_skipped"):
        return f"metrics_skipped=True reason={metrics.get('reason', 'unknown')}"
    parts = []
    for key in _METRIC_PRINT_KEYS:
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_glob", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--sequence_policy",
        default="checkpoint",
        choices=("checkpoint",) + tuple(SEQUENCE_POLICIES),
        help=(
            "checkpoint restores the policy saved during training (legacy checkpoints "
            "fall back to rebuild); rebuild/stored/validate explicitly override it."
        ),
    )
    ap.add_argument("--infer_objective_alpha", type=float, default=None,
                    help="Override objective-delta z-score weight. By default, restore checkpoint action_scoring metadata; legacy checkpoints use the current runtime default.")
    ap.add_argument("--infer_region_alpha", type=float, default=None,
                    help="Override region-prior z-score weight. By default, restore checkpoint action_scoring metadata; legacy checkpoints use the current runtime default.")
    ap.add_argument("--beam_width", type=int, default=1,
                    help=">1 enables objective-aware beam search. 1 uses objective-aware greedy.")
    ap.add_argument("--beam_topk", type=int, default=16,
                    help="Candidate actions expanded per beam step.")
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="Override checkpoint max_tokens for inference context length. Defaults to checkpoint value.")
    ap.add_argument("--no_metrics", dest="return_metrics", action="store_false",
                    help="Disable infer-time layout metric computation.")
    ap.add_argument("--allow_partial_state_dict", action="store_true",
                    help="Allow non-strict checkpoint loading for explicit legacy migration. Defaults to strict loading.")
    ap.add_argument("--no_postprocess", action="store_true",
                    help="Disable deterministic postprocess polish after model rollout.")
    ap.add_argument("--layout_preset", default="checkpoint", choices=sorted(LAYOUT_OBJECTIVE_PRESETS.keys()),
                    help="Inference objective preset: checkpoint keeps stored weights; hpwl/balanced/neat/edge_strict/routeaware override objective knobs.")
    # Optional direct objective overrides.  They are applied after --layout_preset.
    ap.add_argument("--objective_hpwl_weight", type=float, default=None)
    ap.add_argument("--objective_w_hpwl_weight", type=float, default=None)
    ap.add_argument("--objective_nslw_weight", type=float, default=None)
    ap.add_argument("--objective_region_weight", type=float, default=None)
    ap.add_argument("--objective_module_region_weight", type=float, default=None)
    ap.add_argument("--objective_module_floorplan_weight", type=float, default=None)
    ap.add_argument("--module_floorplan_separation_mm", type=float, default=None)
    ap.add_argument("--module_floorplan_overlap_scale", type=float, default=None)
    ap.add_argument("--module_floorplan_channel_scale", type=float, default=None)
    ap.add_argument("--module_floorplan_compact_scale", type=float, default=None)
    ap.add_argument("--module_floorplan_region_scale", type=float, default=None)
    ap.add_argument("--objective_conn_weight", type=float, default=None)
    ap.add_argument("--objective_align_weight", type=float, default=None)
    ap.add_argument("--objective_group_weight", type=float, default=None)
    ap.add_argument("--objective_anchor_weight", type=float, default=None)
    ap.add_argument("--objective_boundary_group_weight", type=float, default=None)
    ap.add_argument("--objective_pitch_weight", type=float, default=None)
    ap.add_argument("--objective_orientation_weight", type=float, default=None)
    ap.add_argument("--objective_density_weight", type=float, default=None)
    ap.add_argument("--objective_soft_spacing_weight", type=float, default=None)
    ap.add_argument("--objective_neatness_weight", type=float, default=None)
    ap.add_argument("--objective_edge_clearance_weight", type=float, default=None)
    ap.add_argument("--objective_interior_weight", type=float, default=None)
    ap.add_argument("--soft_spacing_same_group_extra_mm", type=float, default=None)
    ap.add_argument("--soft_spacing_cross_group_extra_mm", type=float, default=None)
    ap.add_argument("--soft_spacing_large_extra_mm", type=float, default=None)
    ap.add_argument("--same_group_density_scale", type=float, default=None)
    ap.add_argument("--critical_neighbor_density_scale", type=float, default=None)
    ap.add_argument("--anchor_group_density_scale", type=float, default=None)
    ap.add_argument("--large_pair_density_scale", type=float, default=None)
    ap.set_defaults(return_metrics=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    objective_overrides = {
        "hpwl_weight": args.objective_hpwl_weight,
        "w_hpwl_weight": args.objective_w_hpwl_weight,
        "nslw_weight": args.objective_nslw_weight,
        "region_weight": args.objective_region_weight,
        "module_region_weight": args.objective_module_region_weight,
        "module_floorplan_weight": args.objective_module_floorplan_weight,
        "module_floorplan_separation_mm": args.module_floorplan_separation_mm,
        "module_floorplan_overlap_scale": args.module_floorplan_overlap_scale,
        "module_floorplan_channel_scale": args.module_floorplan_channel_scale,
        "module_floorplan_compact_scale": args.module_floorplan_compact_scale,
        "module_floorplan_region_scale": args.module_floorplan_region_scale,
        "conn_weight": args.objective_conn_weight,
        "objective_align_weight": args.objective_align_weight,
        "group_weight": args.objective_group_weight,
        "anchor_weight": args.objective_anchor_weight,
        "boundary_group_weight": args.objective_boundary_group_weight,
        "pitch_weight": args.objective_pitch_weight,
        "orientation_weight": args.objective_orientation_weight,
        "objective_density_weight": args.objective_density_weight,
        "objective_soft_spacing_weight": args.objective_soft_spacing_weight,
        "objective_neatness_weight": args.objective_neatness_weight,
        "objective_edge_clearance_weight": args.objective_edge_clearance_weight,
        "objective_interior_weight": args.objective_interior_weight,
        "soft_spacing_same_group_extra_mm": args.soft_spacing_same_group_extra_mm,
        "soft_spacing_cross_group_extra_mm": args.soft_spacing_cross_group_extra_mm,
        "soft_spacing_large_extra_mm": args.soft_spacing_large_extra_mm,
        "same_group_density_scale": args.same_group_density_scale,
        "critical_neighbor_density_scale": args.critical_neighbor_density_scale,
        "anchor_group_density_scale": args.anchor_group_density_scale,
        "large_pair_density_scale": args.large_pair_density_scale,
    }
    objective_overrides = {k: v for k, v in objective_overrides.items() if v is not None}

    for p in sorted(glob.glob(args.test_glob)):
        res = infer_layout(
            p,
            args.ckpt,
            device=args.device,
            infer_objective_alpha=(
                None if args.infer_objective_alpha is None
                else float(args.infer_objective_alpha)
            ),
            infer_region_alpha=(
                None if args.infer_region_alpha is None
                else float(args.infer_region_alpha)
            ),
            beam_width=int(args.beam_width),
            beam_topk=int(args.beam_topk),
            return_metrics=bool(args.return_metrics),
            max_tokens=args.max_tokens,
            strict_state_dict=not bool(args.allow_partial_state_dict),
            postprocess=not bool(args.no_postprocess),
            layout_preset=str(args.layout_preset),
            objective_overrides=objective_overrides,
            sequence_policy=str(args.sequence_policy),
        )
        out_path = out / (Path(p).stem + ".infer.json")
        save_json(res, out_path)
        metrics_s = _fmt_metrics(res.get("metrics") or {})
        print(
            f"{p} -> {out_path} obj={res['objective']:.3f} "
            f"raw_obj={res.get('objective_raw', 0.0):.3f} "
            f"partial_obj={res.get('objective_partial', 0.0):.3f} "
            f"complete={res.get('complete')} "
            f"placed={res.get('placed_count', 0)}/{res.get('expected_count', 0)} "
            f"failure={res.get('failure_reason')} "
            f"post_delta={res.get('postprocess_objective_delta', 0.0):.3f} "
            f"post_changed={res.get('postprocess_changed_count', 0)} "
            f"postprocess={res.get('postprocess_applied')} "
            f"preset={res.get('layout_preset')} "
            f"mode={res.get('infer_mode')} {metrics_s}"
        )

if __name__ == "__main__":
    main()
