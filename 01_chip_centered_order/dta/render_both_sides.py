import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

def load_files(task_path, layout_path):
    task = json.load(open(task_path, 'r', encoding='utf-8'))
    layout = json.load(open(layout_path, 'r', encoding='utf-8'))
    return task, layout

def build_pad_world_map(task, layout):
    comps = {c.get('name'): c for c in task.get('components', [])}
    pad_world_map = {}  # (comp_name, pad_idx) -> (wx, wy)
    comp_side_map = {}
    for name, entry in layout.items():
        comp = comps.get(name)
        if comp is None:
            continue
        x = float(entry.get('x', 0.0))
        y = float(entry.get('y', 0.0))
        rot_raw = entry.get('rot', 0)
        try:
            rot_val = float(rot_raw)
        except Exception:
            rot_val = 0.0
        rot_deg = rot_val if abs(rot_val) > 4 else rot_val * 90.0
        theta = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        pads = comp.get('pads', []) or comp.get('pad_list', [])
        for pidx, (px, py) in enumerate(pads):
            rx = px * cos_t - py * sin_t
            ry = px * sin_t + py * cos_t
            wx = x + rx
            wy = y + ry
            pad_world_map[(name, pidx)] = (wx, wy)
        comp_side_map[name] = entry.get('side', 'top')
    return pad_world_map, comp_side_map

def plot_side(ax, task, layout, pad_world_map, comp_side_map, side='top'):
    boundary = task.get('boundary', [])
    if boundary:
        poly = Polygon(boundary, closed=True, facecolor='#1b5e20', edgecolor='#0b3d0b', zorder=0)
        ax.add_patch(poly)

    comps = {c.get('name'): c for c in task.get('components', [])}
    pad_radius = 0.6

    # draw components that are on this side
    for name, entry in layout.items():
        s = entry.get('side', 'top')
        if s != side:
            continue
        comp = comps.get(name)
        if comp is None:
            continue
        w = float(comp.get('w', 1.0))
        h = float(comp.get('h', 1.0))
        x = float(entry.get('x', 0.0))
        y = float(entry.get('y', 0.0))
        rot_raw = entry.get('rot', 0)
        try:
            rot_val = float(rot_raw)
        except Exception:
            rot_val = 0.0
        rot_deg = rot_val if abs(rot_val) > 4 else rot_val * 90.0

        rect = Rectangle((x, y), w, h, angle=rot_deg, facecolor='tab:blue', alpha=0.6, edgecolor='k', zorder=2)
        ax.add_patch(rect)

        # draw pads
        pads = comp.get('pads', []) or comp.get('pad_list', [])
        theta = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for pidx, (px, py) in enumerate(pads):
            rx = px * cos_t - py * sin_t
            ry = px * sin_t + py * cos_t
            wx = x + rx
            wy = y + ry
            circ = Circle((wx, wy), radius=max(0.2, min(w, h) * 0.03), facecolor='#d6a447', edgecolor='#8c5d1a', zorder=3)
            ax.add_patch(circ)

    # Draw nets for pads on this side using MST among pads of this side
    import networkx as nx
    for net in task.get('nets', []):
        pts = []
        for cname, pidx in net:
            # include pad only if component placed and on this side
            if cname in comp_side_map and comp_side_map[cname] == side and (cname, pidx) in pad_world_map:
                pts.append(pad_world_map[(cname, pidx)])
        if len(pts) < 2:
            continue
        G = nx.Graph()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                G.add_edge(i, j, weight=d)
        mst = nx.minimum_spanning_tree(G)
        for u, v, d in mst.edges(data=True):
            x1, y1 = pts[u]
            x2, y2 = pts[v]
            ax.plot([x1, x2], [y1, y2], color='#ffeb99', linewidth=0.8, alpha=0.8, zorder=1)

    # set axes limits
    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        pad = max(1.0, 0.05 * max(max(xs) - min(xs), max(ys) - min(ys)))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal')
    ax.invert_yaxis()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--out', default=os.path.join('output', 'replays'))
    args = ap.parse_args()

    task, layout = load_files(args.task, args.layout)
    pad_world_map, comp_side_map = build_pad_world_map(task, layout)

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.task))[0]
    out_path = os.path.join(args.out, f'{base}_both_sides.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
    axes[0].set_title('Top side')
    axes[1].set_title('Bottom side')

    plot_side(axes[0], task, layout, pad_world_map, comp_side_map, side='top')
    plot_side(axes[1], task, layout, pad_world_map, comp_side_map, side='bottom')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print('Saved both-sides image:', out_path)

if __name__ == "__main__":
    main()
