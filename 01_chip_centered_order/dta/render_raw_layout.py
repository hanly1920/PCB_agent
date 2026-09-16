"""Render 'real' PCB images from task + layout JSON.

Usage:
  $env:PYTHONPATH='C:/Users/24663/Desktop/my (2)/my'
  python dta/render_raw_layout.py --task dta/expert_traj (2)/expert_traj/expert1002_traj/expert1002_task.json --layout dta/expert_traj (2)/expert_traj/expert1002_traj/expert1002_layout.json --out output/replays

Saves `raw_<basename>.png` and `raw_<basename>_validation.json` in the output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import networkx as nx


def rotate_point(px, py, w, h, rot):
    # rot: 0,1,2,3 corresponding to 0,90,180,270 clockwise
    if rot == 0:
        return px, py
    elif rot == 1:
        return py, w - px
    elif rot == 2:
        return w - px, h - py
    elif rot == 3:
        return h - py, px
    else:
        return px, py


def parse_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def draw_sample(task_path: Path, layout_path: Path, out_dir: Path):
    task = parse_json(task_path)
    layout = parse_json(layout_path)

    boundary = task.get('boundary')
    components = task.get('components', [])
    nets = task.get('nets', [])

    # compute world bounding box for plotting
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.set_aspect('equal')

    # board polygon
    board_poly = Polygon(boundary, closed=True, facecolor='#0b3d0b', edgecolor='#07320a', linewidth=1.5, zorder=0)
    ax.add_patch(board_poly)

    # draw components
    name_to_comp = {c.get('name'): c for c in components}
    placed_centers = {}
    pad_centers = {}  # (comp_name, pad_idx) -> (x,y)

    for name, pose in layout.items():
        comp = name_to_comp.get(name)
        if comp is None:
            continue
        w = comp.get('w', 1.0)
        h = comp.get('h', 1.0)
        rot = int(round(pose.get('rot', 0) / 90.0)) % 4
        cx = float(pose.get('x', 0.0))
        cy = float(pose.get('y', 0.0))

        # rectangle lower-left corner compute
        # assume pose x,y correspond to top-left or origin? Many expert layouts use component origin at top-left.
        # We'll assume pose x,y is the component's top-left in world coords.
        # For robustness, draw rectangle centered at (cx,cy) with w,h if available otherwise use top-left.

        # Choose to plot rectangle with lower-left at (cx, cy) for consistency with env rendering
        llx = cx
        lly = cy
        rect = Rectangle((llx, lly), w, h, angle=0, facecolor='#f0a040', edgecolor='#120a00', alpha=0.9, zorder=3)
        ax.add_patch(rect)

        # draw pads (convert local pad coords via rotation and offset)
        pad_list = comp.get('pads') or comp.get('pad_list') or []
        for p_idx, pad in enumerate(pad_list):
            px_local, py_local = pad[0], pad[1]
            prx, pry = rotate_point(px_local, py_local, comp.get('w', w), comp.get('h', h), rot)
            wx = llx + prx
            wy = lly + pry
            pad_centers[(name, p_idx)] = (wx, wy)
            circ = Circle((wx, wy), radius=0.5, facecolor='#ffd966', edgecolor='#8c5d1a', zorder=4)
            ax.add_patch(circ)

        placed_centers[name] = (llx, lly, w, h, rot)
        ax.text(llx + w/2.0, lly + h/2.0, name, ha='center', va='center', fontsize=6, zorder=5)

    # draw nets via MST between pad centers
    net_color = '#ffeb99'
    for net in nets:
        pts = []
        for comp_name, pad_idx in net:
            key = (comp_name, pad_idx)
            if key in pad_centers:
                pts.append(pad_centers[key])
        if len(pts) < 2:
            continue
        if len(pts) > 24:
            # fallback: draw pairwise lines (coarse)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=net_color, linewidth=0.6, alpha=0.6, zorder=2)
        else:
            G = nx.Graph()
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                    G.add_edge(i, j, weight=d)
            mst = nx.minimum_spanning_tree(G)
            for u, v, data in mst.edges(data=True):
                x1, y1 = pts[u]
                x2, y2 = pts[v]
                ax.plot([x1, x2], [y1, y2], color=net_color, linewidth=0.8, alpha=0.8, zorder=2)

    ax.set_xlim(min_x - 5, max_x + 5)
    ax.set_ylim(min_y - 5, max_y + 5)
    ax.invert_yaxis()
    ax.axis('off')

    out_png = out_dir / f"raw_{task_path.stem}.png"
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)

    # save simple validation JSON: pad count and sample info
    validation = {
        'name': task_path.stem,
        'num_components': len(components),
        'num_pads': len(pad_centers),
        'components': list(placed_centers.keys())
    }
    out_json = out_dir / f"raw_{task_path.stem}_validation.json"
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    return str(out_png), validation


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    png, val = draw_sample(Path(args.task), Path(args.layout), out_dir)
    print('Wrote', png)
