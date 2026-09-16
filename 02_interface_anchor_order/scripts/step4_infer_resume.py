from __future__ import annotations

import argparse
import glob
import traceback
from pathlib import Path

from pcbplace.infer import infer_layout
from pcbplace.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_glob", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(args.test_glob))

    completed = 0
    skipped = 0
    failed = []

    for index, path in enumerate(paths, start=1):
        src = Path(path)
        out_path = out_dir / (
            src.stem + ".infer.json"
        )

        if out_path.exists() and not args.overwrite:
            print(
                f"[{index}/{len(paths)}] skip existing: "
                f"{out_path}"
            )
            skipped += 1
            continue

        try:
            result = infer_layout(
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

            print(
                f"[{index}/{len(paths)}] FAILED: "
                f"{path}: {exc}"
            )
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
