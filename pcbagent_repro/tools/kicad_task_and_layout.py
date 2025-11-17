#expert_task.json = 这块板子的“问题定义”（有哪些元件、连哪些网、板边界多大）_PCB任务本身（规则+元件+网络），跟谁来下棋无关
#expert_layout.json = 这块板子的“专家布局”（每个元件放在哪、朝哪个方向）——专家真正布出来的版图
#1.从 KiCad 板 → 生成expert_task.json（任务）expert_layout.json（专家布局）
#2.用 expert_task.json 初始化 PCBEnv；
#3.用 expert_layout.json 在 PCBEnv 里重放专家布局 → 得到：专家指标（HPWL、SLW …）专家轨迹 steps, traj；
#4.用 steps 做 行为克隆预训练 TinyDT；
#5.再用 expert_task.json 当作 RL 训练的任务，做 self-play 微调。
import pcbnew
import json
import argparse

def iter_footprints(board):
    """
    兼容 KiCad 6/7/8/9：
    - 新版本推荐用 board.GetFootprints()
    - 某些版本只有 board.Footprints()
    """
    if hasattr(board, "GetFootprints"):
        return board.GetFootprints()
    return board.Footprints()


def extract_task_and_layout(
    kicad_pcb_file,
    grid_N=32,
    min_spacing=0.0,
    hpwl_thresh=10.0,
    b_bonus=1.0,
):
    board = pcbnew.LoadBoard(kicad_pcb_file)

    # ---------- 1) 板边界 -> boundary ----------
    # 用板框的 bounding box，当成 RL 环境里的工作区域
    edges_bbox = board.GetBoardEdgesBoundingBox()
    xmin = pcbnew.ToMM(edges_bbox.GetX())
    ymin = pcbnew.ToMM(edges_bbox.GetY())
    xmax = pcbnew.ToMM(edges_bbox.GetX() + edges_bbox.GetWidth())
    ymax = pcbnew.ToMM(edges_bbox.GetY() + edges_bbox.GetHeight())

    # env.py 里 boundary[0] / boundary[2] 被当作 (xmin,ymin)、(xmax,ymax)
    boundary = [[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]]

    # ---------- 2) 元件信息 ----------
    components = []
    layout = {}
    comp_pad_index = {}

    for fp in iter_footprints(board):
        ref = fp.GetReference()
        bbox = fp.GetBoundingBox()

        # 用 footprint 外接矩形当作元件尺寸
        w = pcbnew.ToMM(bbox.GetWidth())
        h = pcbnew.ToMM(bbox.GetHeight())

        # 以 bbox 左下角作为局部坐标原点（与 env.py 中的逻辑兼容）
        # 注意：KiCad 的 bbox Y 方向是向上的，但我们在整个任务里保持同一坐标系即可
        x0 = pcbnew.ToMM(bbox.GetX())
        y0 = pcbnew.ToMM(bbox.GetY())

        pads = []
        for pad in fp.Pads():
            p_pos = pad.GetPosition()
            px = pcbnew.ToMM(p_pos.x) - x0
            py = pcbnew.ToMM(p_pos.y) - y0
            comp_pad_index[(ref, pad.GetName())] = len(pads)
            pads.append([px, py])

        comp = {
            "name": ref,       # 和 env.comp_index 的 key 对齐
            "w": w,
            "h": h,
            "pads": pads,      # 每个 pad 是 [px, py]，局部坐标
            "type": "normal",  # 先全部标 normal；后续你可以手动改成 chip/core 等
        }
        components.append(comp)
        layout[ref] = {"x": x0, "y": y0, "rot": 0}
    # ---------- 3) 网络：nets ----------
    # env.hpwl / nslw 期望 nets 是：
    #   nets = [
    #       [ [cname, pad_idx], [cname, pad_idx], ... ],  # 一个 net
    #       ...
    #   ]
    net_dict = {}

    for fp in iter_footprints(board):
        ref = fp.GetReference()
        for pad in fp.Pads():
            net_name = pad.GetNetname()
            if not net_name:
                continue
            key = (ref, pad.GetName())
            if key not in comp_pad_index:
                continue
            pidx = comp_pad_index[key]
            net_dict.setdefault(net_name, []).append([ref, pidx])

    nets = [conns for conns in net_dict.values() if len(conns) >= 2]

    task = {
        "grid_N": grid_N,
        "boundary": boundary,
        "min_spacing": min_spacing,
        "hpwl_thresh": hpwl_thresh,
        "b_bonus": b_bonus,
        "components": components,
        "nets": nets,
    }

    return task, layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_task", default=r"D:\777777\pcbagent_repro\pcbagent_repro\expert_traj\expert1_task.json", help="输出 task.json 路径")
    parser.add_argument("--out_layout", default=r"D:\777777\pcbagent_repro\pcbagent_repro\expert_traj\expert1_layout.json")
    parser.add_argument("--grid_N", type=int, default=32)
    parser.add_argument("--min_spacing", type=float, default=0.0)
    parser.add_argument("--hpwl_thresh", type=float, default=10.0)
    parser.add_argument("--b_bonus", type=float, default=1.0)

    # 默认使用本地板文件路径
    parser.add_argument(
        "kicad_pcb",
        nargs="?",
        default=r"D:\777777\pcbagent_repro\pcbagent_repro\tasks\anavi-thermometer.kicad_pcb",
        help="输入的 .kicad_pcb 文件路径",
    )

    args = parser.parse_args()

    task,layout = extract_task_and_layout(
        args.kicad_pcb,
        grid_N=args.grid_N,
        min_spacing=args.min_spacing,
        hpwl_thresh=args.hpwl_thresh,
        b_bonus=args.b_bonus,
    )

    with open(args.out_task, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2)
    with open(args.out_layout, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    print("[kicad_task] Saved task to", args.out_task)
    print("[kicad_task] Saved layout to", args.out_layout)


if __name__ == "__main__":
    main()
