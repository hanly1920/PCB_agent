from __future__ import annotations
import argparse, glob
from pathlib import Path
from pcbplace.infer import infer_layout
from pcbplace.utils import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_glob", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_fast_step", action="store_true", help="Use original env.step() during inference")
    ap.add_argument("--no_postprocess", action="store_true", help="Skip CPU postprocess polishing for fastest inference")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in sorted(glob.glob(args.test_glob)):
        res = infer_layout(
            p,
            args.ckpt,
            device=args.device,
            fast_step=not args.no_fast_step,
            postprocess=not args.no_postprocess,
        )
        out_path = out / (Path(p).stem + ".infer.json")
        save_json(res, out_path)
        print(f"{p} -> {out_path} obj={res['objective']:.3f}")

if __name__ == "__main__":
    main()
