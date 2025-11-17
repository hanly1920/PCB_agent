import argparse, json
from pathlib import Path
from env import PCBEnv

SCRIPT_DIR = Path(__file__).resolve().parent

def rotate_pad(px, py, w, h, rot):
    cx, cy = w/2, h/2
    dx, dy = px - cx, py - cy
    rot = rot % 360
    if rot == 0:   return px, py
    if rot == 90:  return cx + dy, cy - dx
    if rot == 180: return cx - dx, cy - dy
    if rot == 270: return cx - dy, cy + dx
    return px, py

def abs_pad(task, layout, comp_name, pad_idx):
    comp = next((c for c in task["components"] if c["name"] == comp_name), None)
    pl = layout.get(comp_name)
    if comp is None or pl is None or not comp.get("pads") or pad_idx >= len(comp["pads"]):
        return None
    w, h = comp["w"], comp["h"]
    px, py = comp["pads"][pad_idx]
    rx, ry = rotate_pad(px, py, w, h, pl["rot"])
    return (pl["x"] + rx, pl["y"] + ry)

def build_pairs(net, split_multi: bool):
    # 若 split_multi 为 True：将 k 引脚网络扩展为星形对（net [0]，net [i]）
    # 否则：仅接受 2 引脚网络
    if len(net) == 2:
        return [net]
    if len(net) < 2:
        return []
    if not split_multi:
        return []  #忽略多引脚网络
    root = net[0]
    return [[root, other] for other in net[1:]]

def count_nslw(task, layout, boundary, N, tol_cells: float, split_multi: bool):
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    cell_w = (xmax - xmin) / N; cell_h = (ymax - ymin) / N
    tol_abs = max(0.0, tol_cells) * min(cell_w, cell_h)

    cnt = 0
    for net in task.get("nets", []):
        for pair in build_pairs(net, split_multi):
            (c1, p1), (c2, p2) = pair
            a = abs_pad(task, layout, c1, p1)
            b = abs_pad(task, layout, c2, p2)
            if a is None or b is None:
                continue
            horizontal = abs(a[1] - b[1]) <= tol_abs
            vertical   = abs(a[0] - b[0]) <= tol_abs
            if horizontal or vertical:
                cnt += 1
    return cnt

def main():
    ap = argparse.ArgumentParser()
    default_task = (SCRIPT_DIR / "tasks" / "sample_task.json").as_posix()
    default_layout = (SCRIPT_DIR / "out_layout.json").as_posix()
    ap.add_argument("--task", type=str, default=default_task, help=f"Task JSON (default: {default_task})")
    ap.add_argument("--layout", type=str, default=default_layout, help=f"Layout JSON (default: {default_layout})")
    ap.add_argument("--slw_tol_cells", type=float, default=0.5,help="SLW collinearity tolerance in grid cells (0 for strict; default 0.5 cell)")
    ap.add_argument("--no_split_multi", action="store_true",help="Do not split multi-pin nets; only count native 2-pin nets")
    args = ap.parse_args()

    task = json.load(open(args.task, "r"))
    layout = json.load(open(args.layout, "r"))

    #重建环境，以利用其内部逻辑计算 HPWL 和合法性
    env = PCBEnv(task)
    for name, pl in layout.items():
        env.place(name, pl["x"], pl["y"], pl["rot"])

    wl, _ = env.score()
    legal = env.legal_states()

    N = task["grid_N"]
    boundary = task["boundary"]
    nslw_strict = count_nslw(task, layout, boundary, N, tol_cells=0.0, split_multi=False)
    split_multi = (not args.no_split_multi)
    nslw_cfg = count_nslw(task, layout, boundary, N,
                          tol_cells=args.slw_tol_cells, split_multi=split_multi)

    print(f"[eval] HPWL={wl:.2f} "
          f"NSLW_strict={nslw_strict} "
          f"NSLW(tol={args.slw_tol_cells}cell, split_multi={split_multi})={nslw_cfg} "
          f"LegalStates={legal}")

if __name__ == "__main__":
    main()
