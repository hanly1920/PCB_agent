import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from env import PCBPlacementEnv
from DT.models.dt_model import DecisionTransformer


def convert_task_to_env_struct(task: dict, grid_size: int, margin: int = 2):
    xs = [p[0] for p in task.get('boundary', [])]
    ys = [p[1] for p in task.get('boundary', [])]
    world_min_x = min(xs)
    world_min_y = min(ys)
    world_w = max(xs) - world_min_x
    world_h = max(ys) - world_min_y

    usable = grid_size - 2 * margin
    scale = usable / max(world_w, world_h)
    board_w_grid = world_w * scale
    board_h_grid = world_h * scale
    rem_x = grid_size - board_w_grid
    rem_y = grid_size - board_h_grid
    offset_x = rem_x / 2.0
    offset_y = rem_y / 2.0

    def world_to_grid_xy(x: float, y: float):
        gx = (x - world_min_x) * scale + offset_x
        gy = (y - world_min_y) * scale + offset_y
        return gx, gy

    boundary_grid = [tuple(world_to_grid_xy(x, y)) for x, y in task.get('boundary', [])]

    components_raw = task.get('components', [])
    type_to_id = {}
    comp_list = []
    name_to_id = {}
    for comp_id, comp in enumerate(components_raw):
        name = comp.get('name', f'C{comp_id}')
        type_name = comp.get('type', 'default')
        type_id = type_to_id.setdefault(type_name, len(type_to_id))

        w_world = float(comp.get('w', 1.0))
        h_world = float(comp.get('h', 1.0))
        w_cells = max(1, int(round(w_world * scale)))
        h_cells = max(1, int(round(h_world * scale)))

        pad_list_raw = comp.get('pads', []) or comp.get('pad_list', [])
        pad_list = []
        for pad in pad_list_raw:
            px_world, py_world = float(pad[0]), float(pad[1])
            px = int(round(px_world * scale))
            py = int(round(py_world * scale))
            px = min(max(px, 0), max(w_cells - 1, 0))
            py = min(max(py, 0), max(h_cells - 1, 0))
            pad_list.append([px, py])

        entry = {
            'comp_id': comp_id,
            'name': name,
            'type_id': type_id,
            'size': [w_cells, h_cells],
            'pad_list': pad_list
        }
        comp_list.append(entry)
        name_to_id[name] = comp_id

    netlist_indexed = []
    for net in task.get('nets', []):
        processed = []
        for entry in net:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            comp_name, pad_idx = entry
            cid = name_to_id.get(comp_name)
            if cid is None:
                continue
            processed.append((cid, int(pad_idx)))
        if len(processed) >= 1:
            netlist_indexed.append(processed)

    mapping = {
        'world_min_x': world_min_x,
        'world_min_y': world_min_y,
        'world_w': world_w,
        'world_h': world_h,
        'scale': scale,
        'offset_x': offset_x,
        'offset_y': offset_y
    }

    return comp_list, boundary_grid, netlist_indexed, mapping


def grid_to_world(gx: int, gy: int, mapping: dict):
    # inverse of gx = (x - world_min_x)*scale + offset_x
    world_min_x = mapping['world_min_x']
    world_min_y = mapping['world_min_y']
    scale = mapping['scale']
    offset_x = mapping['offset_x']
    offset_y = mapping['offset_y']
    wx = world_min_x + (gx + 0.5 - offset_x) / scale
    wy = world_min_y + (gy + 0.5 - offset_y) / scale
    return float(wx), float(wy)


def build_state_from_obs(obs):
    view = np.asarray(obs['view_mask']).astype(np.float32)
    position = np.asarray(obs['position_mask']).astype(np.float32)
    wire = np.asarray(obs['wire_mask']).astype(np.float32)

    legal_any = (position == 0).any(axis=0).astype(np.float32)
    wire_avg = wire.mean(axis=0).astype(np.float32)
    denom = float(np.max(np.abs(wire_avg)))
    if denom > 1e-6:
        wire_avg = wire_avg / denom
    stacked = np.stack([view, legal_any, wire_avg, legal_any], axis=0)
    return stacked


def run_agent(task_path, layout_path, out_dir, grid_size=64, deterministic=True, max_retries=8):
    VERBOSE_FLAG = bool(int(os.environ.get('VERBOSE_FLAG', '0')))
    task = json.load(open(task_path, 'r', encoding='utf-8'))
    expert_layout = json.load(open(layout_path, 'r', encoding='utf-8'))

    comp_list, boundary_grid, netlist_indexed, mapping = convert_task_to_env_struct(task, grid_size)

    env = PCBPlacementEnv(
        grid_size=grid_size,
        boundary_polygon=boundary_grid,
        component_list=comp_list,
        netlist=netlist_indexed
    )

    device = torch.device('cpu')
    model = DecisionTransformer(grid_size=grid_size, hidden_size=128)
    model.to(device)
    model.eval()

    obs = env.reset()
    done = False

    placements = {}

    # get the placement sequence names to map comp ids to names
    seq = getattr(env, 'placement_sequence', None)
    seq_names = [c.get('name') for c in seq] if seq else [c['name'] for c in comp_list]

    # loop until done; model will sample one action per current component
    step_i = 0
    while not done:
        state = build_state_from_obs(obs)
        states_seq = torch.from_numpy(state).float().unsqueeze(0)
        actions_seq = torch.zeros((1, 3), dtype=torch.long)
        returns_seq = torch.zeros((1, 1), dtype=torch.float32)
        timesteps_seq = torch.zeros((1,), dtype=torch.long)

        with torch.no_grad():
            gx, gy, rot = model.get_action(states_seq, actions_seq, None, returns_seq, timesteps_seq, deterministic=deterministic)

        gx_i, gy_i, rot_i = int(gx), int(gy), int(rot)
        pos_mask = np.asarray(obs['position_mask']).astype(np.float32)
        pgx, pgy, prot = gx_i, gy_i, rot_i
        # find nearest legal using position mask
        # simple search
        found = False
        R, N, _ = pos_mask.shape
        for r in range(R):
            if pos_mask[r].any():
                pass
        # reuse simple nearest search from test script logic
        def find_nearest(pgx, pgy, prot, position_mask, max_radius=8):
            if position_mask is None:
                return pgx, pgy, prot
            R, N, N2 = position_mask.shape
            gx0 = int(max(0, min(N-1, int(round(pgx)))))
            gy0 = int(max(0, min(N2-1, int(round(pgy)))))
            prot0 = int(max(0, min(R-1, int(prot))))
            best = None
            for radius in range(0, max_radius+1):
                for dx in range(-radius, radius+1):
                    dy = radius - abs(dx)
                    for dy_sign in (dy, -dy) if dy != 0 else (dy,):
                        x = gx0 + dx
                        y = gy0 + dy_sign
                        if x < 0 or x >= N or y < 0 or y >= N2:
                            continue
                        for r in range(R):
                            if position_mask[r, y, x] == 0:
                                rot_pref = (r == prot0)
                                rot_diff = 0 if rot_pref else 1
                                score = (rot_diff, abs(dx)+abs(dy_sign))
                                if best is None or score < best[0]:
                                    best = (score, x, y, r)
                if best is not None:
                    _, bx, by, br = best
                    return int(bx), int(by), int(br)
            return gx0, gy0, prot0

        pgx, pgy, prot = find_nearest(gx_i, gy_i, rot_i, pos_mask, max_radius=8)

        # Try the projected action first; if illegal, try alternatives sampled from position_mask
        attempt = 0
        placed_success = False
        last_info = {}
        tried_positions = set()
        while attempt <= max_retries and not placed_success:
            attempt += 1
            x_try, y_try, r_try = int(pgx), int(pgy), int(prot)
            # avoid retrying same
            if (x_try, y_try, r_try) in tried_positions:
                # pick another candidate below
                x_try, y_try, r_try = None, None, None
            try:
                if x_try is not None:
                    obs_ret, reward, done_ret, info = env.step((int(x_try), int(y_try), int(r_try)))
                else:
                    raise Exception('force sample')
            except Exception:
                # fallback: sample from legal positions derived from position_mask
                legal_coords = []
                R, N, N2 = pos_mask.shape
                for r in range(R):
                    ys, xs = np.where(pos_mask[r] == 0)
                    for xi, yi in zip(xs, ys):
                        legal_coords.append((int(xi), int(yi), int(r)))

                if not legal_coords:
                    # no legal positions -> end episode
                    last_info = {'illegal': True, 'reason': 'no_legal_positions'}
                    done = True
                    break

                # score candidates using env estimates (hpwl + slw) if available
                scored = []
                for (xi, yi, ri) in legal_coords:
                    try:
                        # env._estimate_hpwl_if_placed and _estimate_slw_if_placed expect a comp dict
                        comp = env.current_component
                        hpwl_delta = env._estimate_hpwl_if_placed(comp, xi, yi, ri)
                        slw_delta = env._estimate_slw_if_placed(comp, xi, yi, ri)
                        score_val = float(hpwl_delta) + float(slw_delta)
                    except Exception:
                        # fallback to distance-based score
                        score_val = abs(xi - gx_i) + abs(yi - gy_i)
                    scored.append((score_val, xi, yi, ri))

                scored.sort(key=lambda t: t[0])
                pick_idx = min(len(scored)-1, attempt-1)
                _, px, py, pr = scored[pick_idx]
                pgx, pgy, prot = px, py, pr
                tried_positions.add((pgx, pgy, prot))
                # Before calling env.step, validate with env checks to avoid env entering illegal terminal state
                comp_check = env.current_component
                try:
                    w_try, h_try, _ = env._get_rotated_dims_and_pads(comp_check, int(prot))
                except Exception:
                    w_try, h_try = None, None

                if w_try is None:
                    last_info = {'illegal': True, 'reason': 'invalid_rotation_or_comp'}
                    continue

                if not env._check_boundary(int(pgx), int(pgy), w_try, h_try):
                    last_info = {'illegal': True, 'reason': 'boundary_violation_before_step'}
                    # try next candidate
                    continue

                if not env._check_overlap(int(pgx), int(pgy), w_try, h_try, int(prot)):
                    last_info = {'illegal': True, 'reason': 'overlap_violation_before_step'}
                    continue

                if not env._check_spacing(int(pgx), int(pgy), w_try, h_try, int(prot)):
                    last_info = {'illegal': True, 'reason': 'spacing_violation_before_step'}
                    continue

                try:
                    obs_ret, reward, done_ret, info = env.step((int(pgx), int(pgy), int(prot)))
                except Exception as e2:
                    last_info = {'illegal': True, 'reason': 'exception_on_step', 'exc': str(e2)}
                    continue

            # check result
            last_info = info
            if done_ret and info.get('illegal'):
                # illegal: continue trying alternative positions
                placed_success = False
                # if max retries reached, mark done and break
                if attempt > max_retries:
                    done = True
                    break
                else:
                    # prepare to sample next candidate
                    # build legal list again (updated masks)
                    pos_mask = np.asarray(obs['position_mask']).astype(np.float32)
                    # choose next candidate in next loop iteration
                    continue
            else:
                # successful placement or normal done
                placed_success = True
                obs = obs_ret
                reward = reward
                done = done_ret
                info = info

        if not placed_success and last_info.get('illegal'):
            # give up and finish
            if VERBOSE_FLAG:
                print('Giving up after retries, last_info:', last_info)
            break

        # record placement for most recently placed component
        # env.placed_components holds tuples (x,y,w,h,type_id,comp_id)
        # map comp_id -> name
        if VERBOSE_FLAG:
            print(f"Step {step_i}: action chosen (raw) gx={gx},gy={gy},rot={rot} -> projected ({pgx},{pgy},{prot})")
            print('env returned:', 'reward=', reward, 'done=', done, 'info=', info)
            print('current_index now', env.current_index)
        step_i += 1
        for px, py, pw, ph, t_id, comp_id in env.placed_components:
            name = comp_list[comp_id]['name']
            if name not in placements:
                wx, wy = grid_to_world(int(px), int(py), mapping)
                placements[name] = {'x': wx, 'y': wy, 'rot': int(prot), 'side': 'top'}

    # save agent layout
    os.makedirs(out_dir, exist_ok=True)
    agent_layout_path = os.path.join(out_dir, 'agent_layout.json')
    with open(agent_layout_path, 'w', encoding='utf-8') as f:
        json.dump(placements, f, indent=2)

    # render 2x2 figure: expert top,bottom / agent top,bottom
    expert = expert_layout
    agent = placements

    def build_pad_map(local_task, local_layout):
        comps = {c.get('name'): c for c in local_task.get('components', [])}
        pad_world_map = {}
        side_map = {}
        for name, entry in local_layout.items():
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
            side_map[name] = entry.get('side', 'top')
        return pad_world_map, side_map

    exp_pad_map, exp_side_map = build_pad_map(task, expert)
    ag_pad_map, ag_side_map = build_pad_map(task, agent)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)
    axes = axes.flatten()
    titles = ['Expert - Top', 'Expert - Bottom', 'Agent - Top', 'Agent - Bottom']

    def plot_layout(ax, task, layout_map, pad_map, side_map, side):
        boundary = task.get('boundary', [])
        if boundary:
            poly = Polygon(boundary, closed=True, facecolor='#1b5e20', edgecolor='#0b3d0b', zorder=0)
            ax.add_patch(poly)
        comps = {c.get('name'): c for c in task.get('components', [])}
        for name, entry in layout_map.items():
            if entry.get('side', 'top') != side:
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

        if boundary:
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            pad = max(1.0, 0.05 * max(max(xs) - min(xs), max(ys) - min(ys)))
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect('equal')
        ax.invert_yaxis()

    # plot expert top/bottom
    plot_layout(axes[0], task, expert, exp_pad_map, exp_side_map, 'top')
    axes[0].set_title('Expert - Top')
    plot_layout(axes[1], task, expert, exp_pad_map, exp_side_map, 'bottom')
    axes[1].set_title('Expert - Bottom')

    # plot agent top/bottom
    plot_layout(axes[2], task, agent, ag_pad_map, ag_side_map, 'top')
    axes[2].set_title('Agent - Top')
    plot_layout(axes[3], task, agent, ag_pad_map, ag_side_map, 'bottom')
    axes[3].set_title('Agent - Bottom')

    plt.tight_layout()
    out_img = os.path.join(out_dir, 'comparison_expert_agent.png')
    plt.savefig(out_img)
    plt.close(fig)

    print('Saved agent layout to:', agent_layout_path)
    print('Saved comparison image to:', out_img)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--layout', required=True, help='expert layout to compare against')
    ap.add_argument('--out', default=os.path.join('output', 'replays'))
    ap.add_argument('--grid-size', type=int, default=64)
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    run_agent(args.task, args.layout, args.out, grid_size=args.grid_size, deterministic=args.deterministic)
