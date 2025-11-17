#对已有布局进行退火式微调以减少NSLW，同时尽量不增加HPWL
#已有布局：out_layout.json  输出微调后布局：out_layout_refined.json

import argparse, json, math, random
from pathlib import Path
from env import PCBEnv

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
    if len(net) == 2:
        return [net]
    if len(net) < 2:
        return []
    if not split_multi:
        return []
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

#位置落在格线上（左下角对其），而非格心
def to_grid(x, xmin, step):
    return int(round((x - xmin) / step))

def from_grid(g, xmin, step):
    return xmin + (g + 0.5) * step

#目标函数
def objective(task, layout, gamma=50.0, tol_cells=0.0, split_multi=False):
    env = PCBEnv(task)
    for n, pl in layout.items():
        env.place(n, pl["x"], pl["y"], pl["rot"])
    wl, _ = env.score()
    legal = env.legal_states()
    if not all(legal[:3]): #如果布局不合法，返回无穷大目标值（保证这步不会被接受）
        return float("inf"), wl, 0, legal
    nslw = count_nslw(task, layout, task["boundary"], task["grid_N"], tol_cells, split_multi)
    return wl - gamma * nslw, wl, nslw, legal #优化目标：最小化 HPWL - gamma * NSLW

#退火式布局精修
#输入原布局layout，搜索轮数iters，奖励NSLW权重gamma，NSLW容忍度tol_cells，是否拆分多引脚网split_multi，随机种子seed，旋转概率rot_prob
def refine(task, layout, iters=2000, gamma=50.0, tol_cells=0.0, split_multi=False, seed=0, rot_prob=0.1):
    random.seed(seed)
    boundary = task["boundary"]
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    N = task["grid_N"]
    step_x = (xmax - xmin) / N #每格宽度
    step_y = (ymax - ymin) / N #每格高度

    cur = {k: dict(v) for k, v in layout.items()} #传入的layout是dict的dict，复制一份作为当前布局
    best = {k: dict(v) for k, v in layout.items()} #复制一份作为最优布局
    cur_obj, cur_wl, cur_nslw, _ = objective(task, cur, gamma, tol_cells, split_multi)
    best_obj, best_wl, best_nslw = cur_obj, cur_wl, cur_nslw #评估一次cur_obj,cur_wl, cur_nslw作为当前与最优的起点
    #每次迭代随机选择一个元件，随机选择是旋转还是平移（根据rot_prob概率决定），生成候选布局cand，评估其目标值cand_obj
    #根据退火接受准则决定是否接受候选布局为当前布局，若接受则更新cur，若cand_obj优于best_obj则更新最优布局best
    names = list(cur.keys())
    for t in range(1, iters+1):
        name = random.choice(names) #随机选一个元件
        pl = cur[name]
        do_rot = (random.random() < rot_prob)
        cand = {k: dict(v) for k, v in cur.items()}

        if do_rot:
            rot_choices = [0,90,180,270]
            rot_choices.remove(pl["rot"] % 360)
            cand[name]["rot"] = random.choice(rot_choices)
        else:
            gx = to_grid(pl["x"], xmin, step_x)
            gy = to_grid(pl["y"], ymin, step_y)
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            if dx == 0 and dy == 0:
                dx = 1
            ng_x = min(N-1, max(0, gx + dx))
            ng_y = min(N-1, max(0, gy + dy))
            cand[name]["x"] = from_grid(ng_x, xmin, step_x)
            cand[name]["y"] = from_grid(ng_y, ymin, step_y)

        cand_obj, cand_wl, cand_nslw, _ = objective(task, cand, gamma, tol_cells, split_multi)

        # 退火接受准则，更优解必接受，劣解按概率接受
        T = max(0.01, 1.0 - t / max(1.0, iters))
        accept = (cand_obj < cur_obj) or (random.random() < math.exp(-(cand_obj - cur_obj) / T))
        if accept:
            cur, cur_obj, cur_wl, cur_nslw = cand, cand_obj, cand_wl, cand_nslw
            if cand_obj < best_obj:
                best, best_obj, best_wl, best_nslw = cand, cand_obj, cand_wl, cand_nslw

    return best, best_wl, best_nslw

def main():
    # ---- 自动猜项目根并把相对路径转成绝对路径的小工具：优先同目录含 train.py/infer.py，否则用上一级 ----
    def guess_default_proj(script_dir: Path) -> Path: #给定当前脚本所在目录script_dir，尽量找到项目根（比如train.py/infer.py），方便后面把诸如tasks\sample_task.json这种相对路径转绝对路径
        if (script_dir / "train.py").exists() and (script_dir / "infer.py").exists():
            return script_dir #如果当前目录就含train.py/infer.py，就说明它是项目根，直接返回
        if (script_dir.name.lower() == "tools") and (script_dir.parent / "train.py").exists():
            return script_dir.parent #如果当前目录是tools，且上一级含train.py/infer.py，就说明上一级是项目根，返回上一级
        for p in [script_dir.parent, script_dir.parent.parent]:
            if (p / "train.py").exists() and (p / "infer.py").exists():
                return p #兜底再尝试再往上找一层或两层
        return script_dir #实在找不到就返回当前目录

    def abspath(base: Path, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (base / q)

    SCRIPT_DIR = Path(__file__).resolve().parent #__file__为当前脚本路径，.resolve()解析真实路径 .parent取目录部分
    DEFAULT_PROJ = guess_default_proj(SCRIPT_DIR) #返回项目根目录

    ap = argparse.ArgumentParser() #创建一个命令行解释对象，用来定义和解析命令行参数
    #--proj参数：项目根目录，默认值为DEFAULT_PROJ，后续所有相对路径均相对于此目录
    ap.add_argument("--proj", type=str, default=str(DEFAULT_PROJ),help="项目根目录（含 train.py / infer.py / eval_layout.py）")
    #--task参数：任务JSON文件路径，默认值为tasks\sample_task.json
    ap.add_argument("--task", type=str, default=r"tasks\sample_task.json")
    #--in参数：输入布局JSON文件路径，默认值为out_layout.json，修改时修改default值即可
    ap.add_argument("--in", dest="layout_in", type=str, default=r"out_layout.json")
   #--out参数：输出布局JSON文件路径，默认值为out_layout_refined.json
    ap.add_argument("--out", type=str, default=r"out_layout_refined.json")
    #--iters参数：精修迭代次数，默认2000
    ap.add_argument("--iters", type=int, default=6000)
    #--gamma参数：NSLW奖励权重，默认50.0
    ap.add_argument("--gamma", type=float, default=0.99, help="NSLW reward weight")
    #--slw_tol_cells参数：NSLW容忍度，默认0.0
    ap.add_argument("--slw_tol_cells", type=float, default=0.5, help="0=strict")
    #--split_multi参数：布尔标志，指示在计算NSLW时是否拆分多引脚网络，默认False
    ap.add_argument("--split_multi", action="store_true",help="split multi-pin nets when counting NSLW")
    #--seed参数：随机种子，默认0(保证可复现)
    ap.add_argument("--seed", type=int, default=0)
    #--rot_prob参数：旋转概率，默认0.1
    ap.add_argument("--rot_prob", type=float, default=0.3)
    #把命令行输入转成 args 对象，如 args.proj、args.iters 等
    args = ap.parse_args()

    proj = Path(args.proj)
    task_path = abspath(proj, args.task)
    in_path   = abspath(proj, args.layout_in)
    out_path  = abspath(proj, args.out)

    # 读取并执行精修
    task = json.load(open(task_path, "r", encoding="utf-8"))
    layout = json.load(open(in_path, "r", encoding="utf-8"))
    best, wl, nslw = refine( #调用refine函数执行模拟退火微调
        task, layout,
        iters=args.iters,
        gamma=args.gamma,
        tol_cells=args.slw_tol_cells,
        split_multi=args.split_multi,
        seed=args.seed,
        rot_prob=args.rot_prob
    )
    json.dump(best, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False) #把最优结果布局写回到--out输出文件
    print(f"[refine] wrote {out_path}  HPWL={wl:.2f}  NSLW={nslw} (tol={args.slw_tol_cells})")


if __name__ == "__main__":
    main()
