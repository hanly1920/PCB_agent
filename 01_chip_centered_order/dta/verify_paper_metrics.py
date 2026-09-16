import json
import sys
import os
from env import PCBPlacementEnv

def load_layout(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_components_and_nets(layout):
    # Expect layout to contain 'components' list and 'nets' list of lists of (comp_id,pad_idx)
    comps = []
    for c in layout.get('components', []):
        comps.append({
            'comp_id': c.get('comp_id', len(comps)),
            'size': c.get('size', [1,1]),
            'pad_list': c.get('pad_list', []),
            'type_id': c.get('type_id', 0),
            'placed_pads': c.get('placed_pads', [])
        })

    nets = layout.get('nets', [])
    return comps, nets

def main(layout_path):
    layout = load_layout(layout_path)
    comps, nets = build_components_and_nets(layout)

    env = PCBPlacementEnv(grid_size=layout.get('grid_size', 128), component_list=comps, netlist=nets, boundary_polygon=layout.get('boundary'))

    # Ensure env internal structures are initialized
    env.component_list = comps
    env.netlist = nets
    env.load_task()

    # Transfer placed pads from input layout components into env.components
    input_comps = layout.get('components', [])
    for ic in input_comps:
        cid = ic.get('comp_id')
        if cid is None:
            continue
        if cid < len(env.components):
            if 'placed_pads' in ic and ic['placed_pads']:
                env.components[cid]['placed_pads'] = ic['placed_pads']

    # populate placed_components: prefer explicit placed_bboxes, else infer from components' placed_pads
    placed_bboxes = layout.get('placed_bboxes', None)
    env.placed_components = []
    if placed_bboxes:
        env.placed_components = placed_bboxes
    else:
        for c in env.components:
            if 'placed_pads' in c and c['placed_pads']:
                px, py = c['placed_pads'][0]
                w, h = c.get('size', [1,1])
                env.placed_components.append((int(px), int(py), int(w), int(h), int(c.get('type_id',0)), int(c['comp_id'])))

    env._rebuild_masks_from_placed_components()

    # compute new metrics via env
    new_hpwl = env._compute_hpwl()
    new_slw = env._compute_slw()
    new_nslw = getattr(env, '_nslw', 0)
    new_score = env.lambda1 * (1.0 / new_hpwl) + env.lambda2 * new_nslw if new_hpwl>0 else float('inf')

    print('New metrics: HPWL', new_hpwl, 'SLW', new_slw, 'NSLW', new_nslw, 'Score', new_score)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python dta/verify_paper_metrics.py <layout.json>')
        sys.exit(1)
    main(sys.argv[1])
