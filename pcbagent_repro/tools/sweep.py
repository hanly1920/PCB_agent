import argparse, json, subprocess, sys, time, re, shutil
from pathlib import Path
from itertools import product
from copy import deepcopy

def guess_default_proj(script_dir: Path) -> Path:
    """优先选含有 train.py 的目录；否则用上一级。"""
    if (script_dir / "train.py").exists() and (script_dir / "infer.py").exists():
        return script_dir
    if (script_dir.name.lower() == "tools") and (script_dir.parent / "train.py").exists():
        return script_dir.parent
    # 兼容多层嵌套
    for p in [script_dir.parent, script_dir.parent.parent]:
        if (p / "train.py").exists() and (p / "infer.py").exists():
            return p
    return script_dir

def abspath(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def run(cmd, cwd=None):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    return proc.returncode, proc.stdout, dt

def parse_eval_line(s: str):
    # 解析 eval_layout.py 的输出行：
    # [eval] HPWL=1120.37 NSLW_strict=0 NSLW(tol=0.5cell, split_multi=True)=3 LegalStates=(True, True, True, [], [])
    hpwl = None; nslw_strict = None; nslw_cfg = None; legal = None
    m = re.search(r"HPWL=([0-9.]+)", s)
    if m: hpwl = float(m.group(1))
    m = re.search(r"NSLW_strict=(\d+)", s)
    if m: nslw_strict = int(m.group(1))
    m = re.search(r"NSLW\(tol=([0-9.]+)cell, split_multi=(True|False)\)=(\d+)", s)
    if m:
        nslw_cfg = int(m.group(3))
    m = re.search(r"LegalStates=\((True|False), (True|False), (True|False)", s)
    if m:
        legal = tuple(x == "True" for x in m.groups())
    return hpwl, nslw_strict, nslw_cfg, legal


def parse_args():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_PROJ = guess_default_proj(SCRIPT_DIR)

    DEFAULTS = {
        "proj": str(DEFAULT_PROJ),
        "task": r"tasks\sample_task.json",
        "ckpt": r"ckpt.pt",
        "out":  r"results\sweep.csv",

        # 参数网格
        "b_bonus":      [0.01, 0.02, 0.05, 0.1],
        "hpwl_thresh":  [700, 900],
        "slw_tol_cells":[0.5, 0.75],

        # 训练/推理选项
        "train": False,         # True=每个组合都训练一会儿
        "iters": 60,
        "traj_per_iter": 3,

        # 推理后处理/评估口径
        "snap": True,           # 默认启用 --snap_slw
        "no_split_multi": False # 默认拆多端点网计数（更容易看到 NSLW）
    }

    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", type=str, default=DEFAULTS["proj"], help="Project root（包含 train.py / infer.py / eval_layout.py）")
    ap.add_argument("--task", type=str, default=DEFAULTS["task"],help="Task JSON 相对路径（相对 --proj）")
    ap.add_argument("--ckpt", type=str, default=DEFAULTS["ckpt"],help="Checkpoint 路径（相对 --proj）")
    ap.add_argument("--py", type=str, default=sys.executable,help="Python 解释器（默认当前）")

    # 网格默认值
    ap.add_argument("--b_bonus", type=float, nargs="+", default=DEFAULTS["b_bonus"])
    ap.add_argument("--hpwl_thresh", type=float, nargs="+", default=DEFAULTS["hpwl_thresh"])
    ap.add_argument("--slw_tol_cells", type=float, nargs="+", default=DEFAULTS["slw_tol_cells"])

    # 训练与对齐开关
    ap.add_argument("--train", action="store_true", default=DEFAULTS["train"])
    ap.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    ap.add_argument("--traj_per_iter", type=int, default=DEFAULTS["traj_per_iter"])

    # snap 默认开；如需关闭可传 --no_snap
    ap.add_argument("--no_snap", action="store_true", help="关闭对齐后处理（默认开启）")
    ap.add_argument("--no_split_multi", action="store_true", default=DEFAULTS["no_split_multi"],
                    help="不拆多端点网（严格口径）")

    ap.add_argument("--out", type=str, default=DEFAULTS["out"],
                    help="CSV 输出路径（相对 --proj）")
    args = ap.parse_args()

    # 处理 snap 默认开
    args.snap = not args.no_snap
    return args

# --------------- Main ------------------
def main():
    args = parse_args()
    proj = Path(args.proj)
    py = args.py

    train_py = abspath(proj, "train.py")
    infer_py = abspath(proj, "infer.py")
    eval_py  = abspath(proj, "eval_layout.py")
    base_task = abspath(proj, args.task)
    base_ckpt = abspath(proj, args.ckpt)

    out_csv = abspath(proj, args.out)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity
    if not train_py.exists() or not infer_py.exists() or not eval_py.exists():
        print(f"[error] 找不到 train.py/infer.py/eval_layout.py 于 {proj}，请检查 --proj")
        sys.exit(2)
    if not base_task.exists():
        print(f"[error] 找不到任务文件：{base_task}")
        sys.exit(2)
    if (not base_ckpt.exists()) and (not args.train):
        print(f"[warn] {base_ckpt} 不存在，将尝试在每个组合上使用 --train")


    # 读 base task
    base = load_json(base_task)

    # CSV header
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("b_bonus,hpwl_thresh,slw_tol_cells,trained,hpwl,nslw_strict,nslw_cfg,legal1,legal2,legal3,secs_train,secs_infer,secs_eval\n")

    from time import perf_counter
    for bb, ht, tol in product(args.b_bonus, args.hpwl_thresh, args.slw_tol_cells):
        combo_tag = f"bb{bb:+.2f}_ht{ht:.0f}_tol{tol:.2f}"
        print(f"\n=== sweep: {combo_tag} ===")

        # 1) 写临时任务
        task_var = deepcopy(base)
        task_var["b_bonus"] = bb
        task_var["hpwl_thresh"] = ht
        task_path = out_dir / f"task_{combo_tag}.json"
        save_json(task_var, task_path)

        # 2) 训练（可选）
        trained = False
        ckpt_path = base_ckpt
        secs_train = 0.0
        if args.train or (not ckpt_path.exists()):
            ckpt_path = out_dir / f"ckpt_{combo_tag}.pt"
            cmd = [py, str(train_py),
                   "--task", str(task_path),
                   "--iters", str(args.iters),
                   "--traj_per_iter", str(args.traj_per_iter)]
            t0 = perf_counter()
            rc, out, _ = run(cmd, cwd=str(proj))
            secs_train = perf_counter() - t0
            if rc != 0:
                print(f"[train][{combo_tag}] failed:\n{out}")
                continue
            # 如果 train.py 固定写 proj/ckpt.pt，则复制为本组合的 ckpt
            if not ckpt_path.exists():
                default_ckpt = proj / "ckpt.pt"
                if default_ckpt.exists():
                    shutil.copy2(default_ckpt, ckpt_path)
            trained = True

        # 3) 推理
        layout_path = out_dir / f"out_layout_{combo_tag}.json"
        png_path    = out_dir / f"out_layout_{combo_tag}.png"
        cmd = [py, str(infer_py),
               "--task", str(task_path),
               "--ckpt", str(ckpt_path),
               "--out",  str(layout_path),
               "--png",  str(png_path),
               "--slw_tol_cells", str(tol)]
        if args.snap:
            cmd += ["--snap_slw"]
        t0 = perf_counter()
        rc, out, _ = run(cmd, cwd=str(proj))
        secs_infer = perf_counter() - t0
        if rc != 0:
            print(f"[infer][{combo_tag}] failed:\n{out}")
            continue

        # 4) 评估
        cmd = [py, str(eval_py),
               "--task", str(task_path),
               "--layout", str(layout_path),
               "--slw_tol_cells", str(tol)]
        if args.no_split_multi:
            cmd += ["--no_split_multi"]
        t0 = perf_counter()
        rc, out, _ = run(cmd, cwd=str(proj))
        secs_eval = perf_counter() - t0
        if rc != 0:
            print(f"[eval][{combo_tag}] failed:\n{out}")
            continue

        hpwl, nslw_strict, nslw_cfg, legal = parse_eval_line(out)
        legal1, legal2, legal3 = (legal if legal else (None, None, None))

        # 5) 写一行CSV
        with open(out_csv, "a", encoding="utf-8") as f:
            f.write(f"{bb},{ht},{tol},{int(trained)},{hpwl},{nslw_strict},{nslw_cfg},{legal1},{legal2},{legal3},{secs_train:.2f},{secs_infer:.2f},{secs_eval:.2f}\n")

        print(f"[done] {combo_tag}  HPWL={hpwl}  NSLW_strict={nslw_strict}  NSLW_cfg={nslw_cfg}  legal={legal}")

if __name__ == "__main__":
    main()


















# """
# sweep.py — 一键批量跑(train→infer→eval)并生成CSV
#
# 放置位置：建议放到项目根目录或根目录下的 tools/ 目录。
# 直接运行即可（有默认参数）；也可在命令行传参覆盖默认。
#
# 默认网格：
#   b_bonus ∈ {-0.02, -0.05}
#   hpwl_thresh ∈ {700, 900}
#   slw_tol_cells ∈ {0.5, 0.75}
# 默认不训练（复用 ckpt.pt）；若没 ckpt.pt 可传 --train。
#
# PowerShell 示例（留空参数直接跑也行）：
#   "D:\KiCad\9.0\bin\python.exe" sweep.py
# 或：
#   "D:\KiCad\9.0\bin\python.exe" sweep.py --train --iters 60 --traj_per_iter 3
# """
