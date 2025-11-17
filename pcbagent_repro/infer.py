#Algorithm 1（轨迹生成）
import argparse, json, torch
from pathlib import Path
from env import PCBEnv
from seq import chip_oriented_sequence
from model import TinyDT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent #作为默认路径的基准，保证脚本无论从哪里运行都能正确指向项目中的tasks ckpt.pt等

# -------- 几何辅助函数（与env.py保持一致） --------
# 将 (gx, gy) 转为实坐标；与 env.py 的口径保持一致（使用 boundary[0]/[2]）
def grid_to_xy(gx, gy, env, use_center=True):
    xmin, ymin = env.boundary[0]
    xmax, ymax = env.boundary[2]
    W = xmax - xmin
    H = ymax - ymin
    if use_center:
        return xmin + (gx + 0.5) * (W / env.N), ymin + (gy + 0.5) * (H / env.N)
    else:
        return xmin + gx * (W / env.N),       ymin + gy * (H / env.N)

def rotate_size(w, h, rot):
    return (w, h) if (rot % 180 == 0) else (h, w)

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

# -------- 画板边界 器件 引脚 二端网 --------
#画一个空心矩形表示PCB板边界，并在上方标注“BOARD”
def draw_board(ax, boundary):
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False)
    ax.add_patch(rect)
    ax.text((xmin+xmax)/2, ymax, "BOARD", ha='center', va='bottom', fontsize=9)

def draw_components(ax, task, layout, draw_pads=True):
    comp_index = {c["name"]: c for c in task["components"]} #快查器件定义
    for name, pl in layout.items():
        comp = comp_index.get(name)
        if comp is None:  # skip unknown
            continue
        w, h = rotate_size(comp["w"], comp["h"], pl["rot"]) #用rotate_size求出旋转后的宽高
        ax.add_patch(Rectangle((pl["x"], pl["y"]), w, h, fill=False)) #画器件边框
        cx, cy = pl["x"] + w/2, pl["y"] + h/2 #计算器件中心坐标
        ax.text(cx, cy, f"{name}\nrot={pl['rot']}", ha='center', va='center', fontsize=8) #在器件中心标注名称和旋转角度
        if draw_pads and comp.get("pads"):
            for (px, py) in comp["pads"]:
                rx, ry = rotate_pad(px, py, comp["w"], comp["h"], pl["rot"]) #计算旋转后的焊盘坐标
                ax.add_patch(Rectangle((pl["x"]+rx-1.5, pl["y"]+ry-1.5), 3, 3, fill=False)) #画焊盘（3x3的空心方块）

def draw_nets(ax, task, layout, skip_multi=False):
    """Base polyline drawing (for visualization)."""
    comp_index = {c["name"]: c for c in task["components"]}
    for net in task.get("nets", []):
        if skip_multi and len(net) != 2: #只画两引脚网络，跳过多引脚网络
            continue
        pts = []
        ok = True
        for (cname, pidx) in net:
            pl = layout.get(cname); comp = comp_index.get(cname)
            if pl is None or comp is None or not comp.get("pads") or pidx >= len(comp["pads"]):
                ok = False; break
            w, h = comp["w"], comp["h"]
            px, py = comp["pads"][pidx]
            rx, ry = rotate_pad(px, py, w, h, pl["rot"])
            pts.append((pl["x"]+rx, pl["y"]+ry))
        if not ok or len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=1)

# -------- 高亮显示 SLW（二端网）（Straight-Line Wiring） --------
def build_pairs(net, split_multi: bool):
    """Produce 2-pin pairs from a net. If split_multi=False, only return native 2-pin nets."""
    if len(net) == 2:
        return [net]
    if len(net) < 2:
        return []
    if not split_multi:
        return []
    root = net[0]
    return [[root, other] for other in net[1:]]

def highlight_slw_edges(ax, task, layout, boundary, N, tol_cells: float, split_multi: bool):
    """Overlay thicker segments for 2-pin pairs judged as SLW (horizontal/vertical within tolerance)."""
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    cell_w = (xmax - xmin) / N; cell_h = (ymax - ymin) / N
    tol_abs = max(0.0, tol_cells) * min(cell_w, cell_h) #容差：tol_cells个网格大小

    for net in task.get("nets", []):
        for pair in build_pairs(net, split_multi):
            (c1, p1), (c2, p2) = pair
            a = abs_pad(task, layout, c1, p1) #计算第一个焊盘的绝对坐标
            b = abs_pad(task, layout, c2, p2) #计算第二个焊盘的绝对坐标
            if a is None or b is None:
                continue
            horizontal = abs(a[1] - b[1]) <= tol_abs #判断是否水平
            vertical   = abs(a[0] - b[0]) <= tol_abs #判断是否垂直
            if horizontal or vertical:
                ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=2.5)  # 更厚的叠加层（未指定颜色）

def render_png(task, layout, png_path, draw_nets_flag=False, skip_multi=False,
               highlight_slw=False, slw_tol_cells=0.5, split_multi=True):
    boundary = task["boundary"]
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_board(ax, boundary)
    draw_components(ax, task, layout, draw_pads=True)
    if draw_nets_flag:
        draw_nets(ax, task, layout, skip_multi=skip_multi)
    if highlight_slw:
        N = task["grid_N"]
        highlight_slw_edges(ax, task, layout, boundary, N,
                            tol_cells=slw_tol_cells, split_multi=split_multi)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin-10, xmax+10); ax.set_ylim(ymin-10, ymax+10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("PCB Layout Rendering")
    fig.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

# ---------- 将几乎共线的二段连线扣成严格共线 ----------
def _recompute_hpwl_and_legal(task, layout): #将layout放入一个新的环境，重新计算HPWL和合法性
    from env import PCBEnv
    env2 = PCBEnv(task)
    for n, pl in layout.items():
        env2.place(n, pl["x"], pl["y"], pl["rot"])
    wl, _ = env2.score()
    legal = env2.legal_states()
    return wl, legal

def _build_pairs(net, split_multi: bool): #把k端网拆成若干2端网 星形pair
    if len(net) == 2:
        return [net]
    if len(net) < 2 or not split_multi:
        return []
    root = net[0]
    return [[root, other] for other in net[1:]]

def _abs_pad(task, layout, comp_name, pad_idx):
    comp = next((c for c in task["components"] if c["name"] == comp_name), None)
    pl = layout.get(comp_name)
    if comp is None or pl is None or not comp.get("pads") or pad_idx >= len(comp["pads"]):
        return None
    w, h = comp["w"], comp["h"]
    px, py = comp["pads"][pad_idx]
    rx, ry = rotate_pad(px, py, w, h, pl["rot"])
    return (pl["x"] + rx, pl["y"] + ry)

def _comp_type(task, name): #获取元件类型 决定移动优先级
    comp = next((c for c in task["components"] if c["name"] == name), None)
    return comp.get("type") if comp else None

def _inside_boundary(boundary, x, y, w, h): #检查器件放置后是否在边界内
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    return (x >= xmin) and (y >= ymin) and (x + w <= xmax) and (y + h <= ymax)

def _rot_size(w, h, rot):
    return (w, h) if (rot % 180 == 0) else (h, w)

def snap_slw_postprocess(task, layout, slw_tol_cells=0.5, split_multi=True,
                         max_iters=2, accept_worse_hpwl=False):
    # 将“近似共线”的 2-pin 子边，尝试沿轴向微移一个器件，使其严格共线
    boundary = task["boundary"]
    N = task["grid_N"]
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    cell_w = (xmax - xmin) / N; cell_h = (ymax - ymin) / N
    tol_abs = max(0.0, slw_tol_cells) * min(cell_w, cell_h)

    # 基线指标
    base_hpwl, base_legal = _recompute_hpwl_and_legal(task, layout)
    if not all(base_legal[:3]):
        return False, base_hpwl  # 原布局不合法直接返回

    comp_index = {c["name"]: c for c in task["components"]}

    improved = False
    for _ in range(max_iters):
        changed = False
        # 遍历所有网 → 2-pin 子边
        for net in task.get("nets", []):
            for pair in _build_pairs(net, split_multi):
                (c1, p1), (c2, p2) = pair
                a = _abs_pad(task, layout, c1, p1)
                b = _abs_pad(task, layout, c2, p2)
                if a is None or b is None:
                    continue
                dx = b[0] - a[0]; dy = b[1] - a[1]
                # 已严格共线则跳过
                if abs(dy) <= 1e-9 or abs(dx) <= 1e-9:
                    continue
                # 近似共线——决定朝哪个轴对齐
                horizontal = abs(dy) <= tol_abs
                vertical   = abs(dx) <= tol_abs
                if not (horizontal or vertical):
                    continue

                # 选移动的器件：优先移动非芯片；相同则移动面积小的（芯片通常更重）
                move_name, stay_name = c2, c1
                t1, t2 = _comp_type(task, c1), _comp_type(task, c2)
                if (t2 == "chip") and (t1 != "chip"):
                    move_name, stay_name = c1, c2
                elif t1 != t2 and t1 == "chip":
                    move_name, stay_name = c2, c1
                else:
                    # 比一下面积
                    A = comp_index[c1]; B = comp_index[c2]
                    area1 = A["w"] * A["h"]; area2 = B["w"] * B["h"]
                    if area1 < area2:
                        move_name, stay_name = c1, c2

                # 计算“把 move_name 扣到严格共线”后的新 (x,y)
                pl_m = layout[move_name]; pl_s = layout[stay_name]
                comp_m = comp_index[move_name]
                w_m, h_m = _rot_size(comp_m["w"], comp_m["h"], pl_m["rot"])
                new_x, new_y = pl_m["x"], pl_m["y"]
                # 先算两边 pad 的绝对坐标（再用差值得到应移动量）
                pa = _abs_pad(task, layout, c1, p1)
                pb = _abs_pad(task, layout, c2, p2)
                if pa is None or pb is None:
                    continue

                if horizontal:
                    # 水平对齐：只改y
                    target_y = pb[1] if move_name == c1 else pa[1]
                    delta = target_y - (pa[1] if move_name == c1 else pb[1])
                    new_y = pl_m["y"] + delta
                elif vertical:
                    # 扣到同一直线：把 x 对齐
                    target_x = pb[0] if move_name == c1 else pa[0]
                    delta = target_x - (pa[0] if move_name == c1 else pb[0])
                    new_x = pl_m["x"] + delta

                # 边界检查
                if not _inside_boundary(boundary, new_x, new_y, w_m, h_m):
                    continue

                # 申请试探：复制 layout，替换一个器件的位置
                cand = {k: dict(v) for k, v in layout.items()}
                cand[move_name]["x"] = new_x
                cand[move_name]["y"] = new_y

                cand_hpwl, cand_legal = _recompute_hpwl_and_legal(task, cand)
                if all(cand_legal[:3]) and (accept_worse_hpwl or cand_hpwl <= base_hpwl + 1e-6):
                    # 接受改动
                    layout[move_name]["x"] = new_x
                    layout[move_name]["y"] = new_y
                    base_hpwl = cand_hpwl
                    changed = True
                    improved = True
        if not changed:
            break

    return improved, base_hpwl

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    default_task = (SCRIPT_DIR / "expert_traj" / "expert1_task.json").as_posix()
    default_ckpt = (SCRIPT_DIR / "ckpt_bc.pt").as_posix()
    default_out  = (SCRIPT_DIR / "out_layout.json").as_posix()
    default_png  = (SCRIPT_DIR / "out_layout.png").as_posix()
    #action="store_true" 的规则是——不写这个参数时取 False，写了这个参数时取 True
    ap.add_argument("--task", type=str, default=default_task, help=f"Task JSON (default: {default_task})")
    ap.add_argument("--ckpt", type=str, default=default_ckpt, help=f"Checkpoint (default: {default_ckpt})")
    ap.add_argument("--out",  type=str, default=default_out,  help=f"Output layout JSON (default: {default_out})")
    ap.add_argument("--png",  type=str, default=default_png,  help=f"Output PNG (default: {default_png})")

    ap.add_argument("--no_draw_nets", action="store_true",help="Do NOT draw nets polylines (default: draw)") #不画折线（默认画），只花器件和边界
    ap.add_argument("--skip_multi", action="store_true",help="Skip multi-pin nets when drawing polylines") #画折线时跳过多引脚网络，打开它=更快，信息更少；关闭=更完整
    ap.add_argument("--no_highlight_slw", action="store_true",help="Do NOT highlight SLW (default: highlight)") #不高亮SLW（默认高亮）
    ap.add_argument("--slw_tol_cells", type=float, default=0.5,help="SLW collinearity tolerance in grid cells (0 for strict; default 0.5 cell)") #SLW容差，单位网格（0为严格，默认0.5个网格），更小=更严格（更容易报冲突、NSLW 更难通过），更大=更宽松（不太报冲突）
    ap.add_argument("--no_split_multi", action="store_true",help="Do not split multi-pin nets when highlighting SLW") #高亮SLW时不拆分多引脚网络，打开它=少画线但可能漏掉细节；关闭=更细致

    # SLW snapping controls (post-process)
    ap.add_argument("--snap_slw", action="store_true",help="Snap near-collinear 2-pin pairs onto strict collinearity if legal.") #对近共线的两点做微调，尽量对齐成严格共线（若合法）。打开它=NSLW 更容易过；可能对 HPWL 有轻微影响。
    ap.add_argument("--snap_max_iters", type=int, default=2,help="Max sweep passes for snapping (default 2).")#对齐的扫描轮数。更大=更彻底，时间↑，收益递减；更小=更快，可能不够彻底
    ap.add_argument("--snap_accept_worse_hpwl", action="store_true",help="Allow HPWL increase when snapping (default: disallow).") #是否允许为了 NSLW 而牺牲 HPWL。默认不允许（稳妥）。只在你必须过 NSLW 时考虑打开。

    ap.add_argument("--temperature", type=float, default=0.05) #联合 softmax 的温度。更小=更“贪心”（稳定、少探索）；更大=更“活跃”（易踩坑），推理推荐 0.1~0.3；需要探索时可到 0.4~0.6。
    ap.add_argument("--rots", type=str, default="0,90,270")  # 先用你原来的“3N²”，更多朝向=更大搜索空间（潜在更优，但不稳）；更少=稳。
    ap.add_argument("--hpwl_thresh", type=float, default=900.0,help="Hard-ban grid cells whose HPWL delta exceeds this threshold.") # HPWL 增量阈值，超过视为非法位置。更小=更严格（易稳）；更大=更宽松（易踩坑）。可根据任务规模调整。

    ap.add_argument("--topk", type=int, default=256, help="Sample only from joint top-k logits.") #只在联合 logits 的前 k 名里采样。更小=更像贪心、很稳但可能错过好解；更大=更探索、起伏大。
    ap.add_argument("--wire_lambda", type=float, default=0.5, help="Soft penalty strength for wire cost.") #惩罚强度（从 logits 里减去 λ * penalty）。更大=更讨厌高 wire（HPWL 往往更低，但可能早收缩）；更小=更宽容（探索更多，结果起伏可能变大）。推荐：0.3~0.5；想更激进可到 0.6~0.8。
    ap.add_argument("--wire_q", type=float, default=0.98, help="Percentile for wire normalization (0-1).") #用合法格子的 wire 第 q 分位数做归一化尺度（自适应不同板子/步的数值范围）。更高（→尺度更大）=同样的 wire 惩罚更弱（更宽松）；更低（→尺度更小）=惩罚更强。推荐：0.90~0.98。想更保守可提到 0.98。

    args = ap.parse_args()

    task_path = Path(args.task); ckpt_path = Path(args.ckpt); out_path = Path(args.out); png_path = Path(args.png)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}  (先运行 train.py 生成 ckpt.pt)")

    #生成layout
    task = json.load(open(task_path, "r"))
    env  = PCBEnv(task)
    model = TinyDT(task["grid_N"])
    model.load_state_dict(torch.load(ckpt_path.as_posix(), map_location="cpu"))

    seq = chip_oriented_sequence(task)
    layout = {}
    allowed_rots = tuple(int(x) for x in args.rots.split(","))
    for name in seq:
        tokens = env.state_token(name)
        # === 关键改动：把 wire 超阈值并入硬掩码 position，避免抽到极差格子 ===
        for r, t in tokens.items():
            p = np.asarray(t["position"]);w = np.asarray(t["wire"]);b = np.asarray(t["bonus"])
            legal_mask = p < 0.5
            n_legal = int(legal_mask.sum())

            if n_legal > 0:
                w_legal = w[legal_mask]
                b_legal = b[legal_mask]
                w_stats = np.percentile(w_legal, [50, 95]).tolist() + [float(w_legal.max())]
                b_stats = np.percentile(b_legal, [50, 95]).tolist() + [float(b_legal.max())]
            else:
                # 没有合法格，就不要算 percentile 了，防止报错
                w_stats = ["NA", "NA", "NA"]
                b_stats = ["NA", "NA", "NA"]

            print(
                f"[dbg] {name} rot={r} pos#legal={n_legal} "
                f"wire[p50,p95,max]={w_stats} "
                f"bonus[p50,p95,max]={b_stats}"
            )
            pos = np.asarray(t["position"], dtype=np.float32)  # 1=非法, 0=合法
            wire = np.asarray(t["wire"], dtype=np.float32)  # HPWL 增量 / 截断后的代价
            wire_mask = (wire > args.hpwl_thresh).astype(np.float32)  # 超阈值 -> 视为非法
            t["position"] = np.maximum(pos, wire_mask)  # 合并到 position 硬掩码

        # 联合分布抽样 (R×N×N)
        gx, gy, rot = model.sample_action(tokens, temperature=args.temperature, allowed_rots=allowed_rots,topk=args.topk,wire_lambda=args.wire_lambda,wire_q=args.wire_q,)

        # 落子（格心坐标，和 bonus 口径一致）
        x, y = grid_to_xy(gx, gy, env, use_center=True)
        env.place(name, x, y, rot)
        layout[name] = {"x": x, "y": y, "rot": rot}

    json.dump(layout, open(out_path, "w"), indent=2)
    print("[infer] wrote", out_path)
    # ---- SLW snapping post-process ----
    if args.snap_slw:
        improved, new_hpwl = snap_slw_postprocess(
            task, layout,
            slw_tol_cells=args.slw_tol_cells,
            split_multi=(not args.no_split_multi),
            max_iters=args.snap_max_iters,
            accept_worse_hpwl=args.snap_accept_worse_hpwl,
        )
        if improved:
            # 覆盖保存对齐后的布局
            json.dump(layout, open(out_path, "w"), indent=2)
            print(f"[snap] applied. new HPWL={new_hpwl:.2f}")

    draw_nets_flag = not args.no_draw_nets
    highlight_slw_flag = not args.no_highlight_slw
    split_multi_flag = not args.no_split_multi
    render_png(
        task, layout, png_path=png_path.as_posix(),
        draw_nets_flag=draw_nets_flag,
        skip_multi=args.skip_multi,
        highlight_slw=highlight_slw_flag,
        slw_tol_cells=args.slw_tol_cells,
        split_multi=split_multi_flag,
    )

    print("[render] wrote", png_path)

    # sd = torch.load(args.ckpt, map_location="cpu") #加载模型参数
    # model.load_state_dict(sd); print("[infer] loaded", args.ckpt, "params", sum(p.numel() for p in model.parameters())) #打印参数总数
    print("[infer] model.N =", model.N, "task.grid_N =", task["grid_N"])


if __name__ == "__main__":
    main()
