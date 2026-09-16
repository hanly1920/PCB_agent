import json
import numpy as np
from dta.agent_rollout_and_render import convert_task_to_env_struct, grid_to_world
from env import PCBPlacementEnv


def diag(task_path, layout_path, grid_size=64):
    task = json.load(open(task_path, 'r', encoding='utf-8'))
    layout = json.load(open(layout_path, 'r', encoding='utf-8'))
    comp_list, boundary_grid, netlist_indexed, mapping = convert_task_to_env_struct(task, grid_size)
    env = PCBPlacementEnv(grid_size=grid_size, boundary_polygon=boundary_grid, component_list=comp_list, netlist=netlist_indexed)
    obs = env.reset()

    print('Number of components in env:', env.num_components)
    print('Placement sequence names:')
    for i, c in enumerate(env.placement_sequence):
        print(i, c.get('name'), 'type', c.get('type_id'), 'size', c.get('size'))

    # For each component at reset, compute legal positions count
    for idx, comp in enumerate(env.placement_sequence):
        pm = env.build_position_mask(comp)
        legal = np.sum(pm == 0)
        print(f'Component {idx} {comp.get("name")} legal positions:', int(legal))

    # Place expert U1 if present in layout
    if 'U1' in layout:
        u1 = layout['U1']
        print('\nPlacing expert U1 at', u1)
        # convert world to grid
        # find nearest grid integer by using env coordinate system: our env expects integer grid coords
        gx = int(round(u1['x']))
        gy = int(round(u1['y']))
        rot = int(u1.get('rot', 0))
        try:
            # directly call env._place_component to mirror env.step placement
            comp = env.current_component
            w, h, _ = env._get_rotated_dims_and_pads(comp, rot)
            env._place_component(gx, gy, w, h, rot, comp['comp_id'])
            env._rebuild_masks_from_placed_components()
        except Exception as e:
            print('Exception placing U1:', e)

    print('\nAfter placing U1, recompute legal positions for remaining components:')
    for idx in range(env.current_index, env.num_components):
        comp = env.placement_sequence[idx]
        pm = env.build_position_mask(comp)
        legal = np.sum(pm == 0)
        print(f'Component {idx} {comp.get("name")} legal positions after U1:', int(legal))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--layout', required=True)
    ap.add_argument('--grid-size', type=int, default=64)
    args = ap.parse_args()
    diag(args.task, args.layout, grid_size=args.grid_size)
