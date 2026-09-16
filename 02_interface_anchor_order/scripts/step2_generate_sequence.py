from __future__ import annotations

import argparse

from pcbplace.utils import load_json, save_json
from pcbplace.sequence_heuristic import generate_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--bfs_depth", type=int, default=3)
    args = ap.parse_args()

    data = load_json(args.in_json)
    seq, meta = generate_sequence(data, bfs_depth=args.bfs_depth, return_meta=True)
    data.setdefault("graph", {})
    data["graph"]["sequence"] = seq
    data["graph"]["sequence_source"] = "heuristic_anchor_interface_priority_v6"
    data["graph"]["sequence_config"] = meta
    save_json(data, args.out_json)
    print(f"Wrote: {args.out_json} (len={len(seq)}) source={data['graph']['sequence_source']}")


if __name__ == "__main__":
    main()
