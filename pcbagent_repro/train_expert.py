import argparse, json, torch
from env import PCBEnv
from seq import chip_oriented_sequence
from model import TinyDT
from replay import PrioritizedReplay
import torch
import numpy as np
import torch.optim as optim

# 将 (gx, gy) 转为实坐标；与 env.py 的口径保持一致（使用 boundary[0]/[2]）
def grid_to_xy(gx, gy, env):
    xmin, ymin = env.boundary[0]
    xmax, ymax = env.boundary[2]
    W = xmax - xmin; H = ymax - ymin
    return xmin + (gx + 0.5) * (W / env.N), ymin + (gy + 0.5) * (H / env.N)

def xy_to_grid(x, y, env):
    xmin, ymin = env.boundary[0]
    xmax, ymax = env.boundary[2]
    W = xmax - xmin
    H = ymax - ymin
    gx = int((x - xmin) / W * env.N)
    gy = int((y - ymin) / H * env.N)
    # 防止超界
    gx = max(0, min(env.N - 1, gx))
    gy = max(0, min(env.N - 1, gy))
    return gx, gy

# 统一的打分函数：根据 HPWL(wl) 和 SLW(slw) 计算 score
def compute_score(wl, slw, lam1=1000.0, lam2=0.1):
    """
    wl   : HPWL（越小越好）
    slw  : 可直线布线的二端网数量（越大越好）
    lam1 : 线长的权重
    lam2 : SLW 的权重

    返回值 score 越大代表轨迹越好。
    """
    return lam1 * (1.0 / max(wl, 1e-6)) + lam2 * slw

def build_expert_traj(task, layout): #把“专家布局 JSON”转换为 TinyDT 能学习的「离线 demonstrations」
    """
    task:  expert_task.json 读出来的 dict
    layout: expert_layout.json 读出来的 dict
    返回：
      steps: [(tokens, (gx,gy,rot)), ...]
      traj:  [(name, x, y, rot), ...]
    """
    env = PCBEnv(task) #创建环境
    env.reset() #清空布局
    seq = chip_oriented_sequence(task) #获取元件放置顺序

    steps = []
    traj = []

    for name in seq:
        tokens = env.state_token(name)  #从 env 里拿到当前状态，这会为这个元件生成四个朝向下的 (position, wire, bonus) 三通道图。与自博弈时完全一致
        pl = layout.get(name) #从专家布局里拿到该元件的放置信息
        if pl is None:  # 可能有一些标记件不在 layout 中，就 continue 跳过。
            continue

        gx, gy = xy_to_grid(pl["x"], pl["y"], env)
        rot = pl.get("rot", 0)

        # 训练用的数据格式
        steps.append((tokens, (gx, gy, rot))) #这就是bc里用的状态-动作对
        # 同时在 env 里真的 place(name, x, y, rot)，把专家解逐个写进去，这样最后可以算一遍整体 HPWL/SLW。
        env.place(name, pl["x"], pl["y"], rot)
        traj.append((name, pl["x"], pl["y"], rot))

    wl, slw = env.score()   # 看一下专家板子的指标
    print(f"[expert] HPWL={wl:.2f}, SLW={slw}")
    return steps, traj, (wl, slw)#返回steps: 专家轨迹的「训练视角」：每步是 (tokens, (gx,gy,rot))。traj: 专家轨迹的「布局视角」：每步是 (name, x, y, rot)。

#专家行为克隆
def pretrain_on_expert(expert_task_path, expert_layout_path,epochs=10, lr=1e-3, device="cpu"):
    # 1 读专家 task/layout
    with open(expert_task_path, "r", encoding="utf-8") as f:
        task = json.load(f)
    with open(expert_layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    # 2 用专家 task 构造 TinyDT，保证 N 一致
    N = task["grid_N"] #从已经读进内存的任务字典 task 里，取出键 "grid_N" 对应的整数
    model = TinyDT(N).to(device)

    # 3 构造专家轨迹
    steps, traj, (wl, slw) = build_expert_traj(task, layout)
    print(f"[expert] HPWL={wl:.2f}, SLW={slw}")
    print(f"[bc] expert traj len={len(steps)}")

    # 4 行为克隆预训练
    opt = optim.Adam(model.parameters(), lr=lr) #优化器Adam
    for ep in range(epochs):
        loss = traj_loss(model, steps, mask_illegal=False)#就是让模型在每个 (tokens, (gx,gy,rot)) 上，尽可能赋予专家动作更高概率
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[bc] epoch={ep} loss={loss.item():.4f}")
    print("[bc] finished pretrain")
    return model, task  # 返回已经做过若干 epoch BC 的 model以及 task

#给定放置顺序seq，让策略模型model在环境env里逐渐放置元件，生成一条轨迹traj及其得分score，供上层在线微调调用
def trajectory_generation(env, model, seq): #轨迹生成
    traj = [] #轨迹容器，是一个元件放置列表，元素格式为 (name, x, y, rot)
    for name in seq:
        tokens = env.state_token(name) #为下一个元件生成四个朝向的三通道图
        gx, gy, rot = model.sample_action(tokens) #在四个朝向里各自采样一个格点，选择概率最高的朝向和位置
        xmin,ymin = env.boundary[0] #取第一个点
        xmax,ymax = env.boundary[2] #取第三个点
        #boundary 是一个四点列表，分别是左下、右下、右上、左上
        W = xmax-xmin; H = ymax-ymin
        x = xmin + (gx + 0.5) * (W / env.N)
        y = ymin + (gy + 0.5) * (H / env.N)
        env.place(name, x, y, rot) #在环境中放置该元件
        traj.append((name, x, y, rot))
    wl, slw = env.score()
    score = compute_score(wl, slw, lam1=1000.0, lam2=0.01)
    return traj, score

#输入一条轨迹traj_steps（是一条完整轨迹中每一步的状态、动作序列），计算其负对数概率损失，用于微调 TinyDT 模型
def traj_loss(model, traj_steps, normalize_w= True, mask_illegal: bool = True):# 计算轨迹的负对数概率损失
    logps = []
    for tokens, (gx,gy,rot) in traj_steps:
        logits_by_rot = model(tokens, mask_illegal=mask_illegal)               # {rot: N*N logits}
        idx = gy * model.N + gx                     # 展平索引
        logits = logits_by_rot[rot]                 # 这一步对应的朝向下，对所有格点的logits
        logp = torch.log_softmax(logits, dim=0)[idx] # 该动作的 log 概率
        logps.append(logp)
    return -torch.stack(logps).sum() #把每步的 log 概率相加，取负号作为负对数似然 (NLL)，最小化它等价于让模型更倾向于复现这条轨迹里的动作（即行为克隆/最大似然）
#task 是 PCB 布局任务的描述字典； iters 是微调轮数； traj_per_iter 是每轮收集的合法轨迹数； batch_size 是每轮训练批次大小； lr 是学习率
'''def online_finetune(task, iters=10, traj_per_iter=100, batch_size=8, lr=1e-3, lam1=1000.0, lam2=0.1, model=None):
    env = PCBEnv(task)
    if model is None:  # 只有在调用者没给model时，让这个函数既能“自己建模型”，也能“用别人传进来的现成模型”。
        model = TinyDT(task["grid_N"])
    opt = torch.optim.Adam(model.parameters(), lr=lr) #Adam 优化器
    buffer = PrioritizedReplay(alpha=1.0) #优先回放池，用于存储轨迹和分数，alpha=1.0 表示优先级完全按分数区分（越大越容易被采样）
    total_collected = 0
    N = task["grid_N"]
    for it in range(iters):
        collected = []
        while len(collected) < traj_per_iter:
            env.reset()
            seq = chip_oriented_sequence(task)

            #tokens：该器件四个朝向的三通道图（position / wire / bonus）
            traj = []; steps = []
            for name in seq: #让当前策略自博弈，探索动作空间，把好的轨迹存入经验池。
                tokens = env.state_token(name)
                gx, gy, rot = model.sample_action(tokens) #在四个朝向里各自采样一个格点，选择概率最高的朝向和位置
                x, y = grid_to_xy(gx, gy, env)
                env.place(name, x, y, rot)
                traj.append((name, x, y, rot))
                steps.append((tokens, (gx,gy,rot))) #记录每步的 (tokens, action) 以便后续计算损失

            wl, slw = env.score()
            boundary_ok, spacing_ok, all_ok, *_ = env.legal_states()
            if boundary_ok and spacing_ok and all_ok:
                score = compute_score(wl, slw, lam1=lam1, lam2=lam2)
                buffer.add((steps, traj), score) #存到buffer，并放进本轮的collected用于当次训练
                collected.append((steps, score)) #记录在本轮收集列表中
        total_collected += len(collected)
        print(f"[iter {it}] collected={len(collected)}  total={total_collected}  buffer={len(buffer)}")

        # ---- 训练 ----
        batch_items = buffer.topk(batch_size, with_scores=True)  # 从优先回放池中取出按优先级排序的前 k 条轨迹。
        # payload 是你 add 时塞进去的 (steps, traj)
        steps_list = []
        scores = []
        for (payload, sc) in batch_items:
            steps, _traj = payload
            steps_list.append(steps) #多个轨迹的steps列表
            scores.append(float(sc))

        scores = np.array(scores, dtype=np.float32)

        # 对 scores 做一个归一化 + 非线性，得到权重 w
        w = (scores - scores.mean()) / (scores.std() + 1e-6)
        w = np.clip(w, -2.0, 2.0)
        w = np.exp(w) #指数：exp 把差异放大（好轨迹的权重大）
        w = w / (w.mean() + 1e-6)

        opt.zero_grad()
        loss = 0.0
        for steps, wi in zip(steps_list, w):
            loss = loss + (traj_loss(model, steps) * float(wi))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    #训练结束后，让模型再放一遍板，拿一条最终轨迹 traj
    env.reset()
    seq = chip_oriented_sequence(task)
    traj, _ = trajectory_generation(env, model, seq)
    return model, traj'''

def online_finetune(
    task,
    iters=10,
    traj_per_iter=100,
    batch_size=8,
    lr=1e-3,
    lam1=1000.0,
    lam2=0.1,
    model=None,
):
    """
    用于“专家预训练后的 RL 微调”的 online_finetune 版本：
    - 如果传入了 model（BC 后的模型），就在这个基础上继续训练；
    - 仍然使用 PrioritizedReplay.add + topk，不引入 sample；
    - 加入 attempts 上限 + dbg 打印，避免在难搜到合法轨迹时看起来像卡死。
    """
    env = PCBEnv(task)
    if model is None:
        model = TinyDT(task["grid_N"])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = PrioritizedReplay(alpha=1.0)
    total_collected = 0

    for it in range(iters):
        collected = []
        attempts = 0
        max_attempts = traj_per_iter * 50  # 每轮最多尝试这么多局，防止死循环

        while len(collected) < traj_per_iter and attempts < max_attempts:
            attempts += 1
            env.reset()
            seq = chip_oriented_sequence(task)

            traj = []
            steps = []

            # 让当前策略自博弈放完一板
            for name in seq:
                tokens = env.state_token(name)
                gx, gy, rot = model.sample_action(tokens)
                x, y = grid_to_xy(gx, gy, env)
                env.place(name, x, y, rot)
                traj.append((name, x, y, rot))
                steps.append((tokens, (gx, gy, rot)))

            wl, slw = env.score()
            # 根据 eval_layout 的输出推断：返回 (boundary_ok, spacing_ok, all_ok, viol_comps, viol_pairs)
            boundary_ok, spacing_ok, all_ok, viol_comps, viol_pairs = env.legal_states()

            # 每 10 次尝试打印一次采样情况，方便调试
            if attempts % 10 == 0:
                print(
                    f"[dbg-RL] it={it} try={attempts} "
                    f"collected={len(collected)} "
                    f"boundary_ok={boundary_ok} spacing_ok={spacing_ok} all_ok={all_ok} "
                    f"HPWL={wl:.2f} SLW={slw} "
                    f"viol_comps={viol_comps} viol_pairs={viol_pairs}"
                )

            # 这里是“认可一条轨迹”的条件：
            # 现在用 all_ok；如果太难收到，可以临时改成 boundary_ok and spacing_ok
            if all_ok:
                score = compute_score(wl, slw, lam1=lam1, lam2=lam2)
                buffer.add((steps, traj), score)
                collected.append((steps, score))
                total_collected += 1

        print(
            f"[iter {it}] attempts={attempts} "
            f"collected={len(collected)} total={total_collected} buffer={len(buffer)}"
        )

        if len(buffer) == 0:
            print("[warn] buffer empty, skip training this iter")
            continue

        # ---- 训练：和你之前一样，从 buffer.topk 取一批轨迹做加权 BC ----
        batch_items = buffer.topk(min(batch_size, len(buffer)), with_scores=True)
        steps_list = []
        scores = []
        for (payload, sc) in batch_items:
            steps, _traj = payload
            steps_list.append(steps)
            scores.append(float(sc))

        scores = np.array(scores, dtype=np.float32)

        # 对 scores 做归一化 → 权重 w
        w = (scores - scores.mean()) / (scores.std() + 1e-6)
        w = np.clip(w, -2.0, 2.0)
        w = np.exp(w)
        w = w / (w.mean() + 1e-6)

        opt.zero_grad()
        loss = 0.0
        for steps, wi in zip(steps_list, w):
            loss = loss + traj_loss(model, steps) * float(wi)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        print(f"[train-RL] iter={it} loss={loss.item():.4f}")

    # 训练结束后，让模型再放一遍板，拿一条最终轨迹 traj
    env.reset()
    seq = chip_oriented_sequence(task)
    traj, _ = trajectory_generation(env, model, seq)
    return model, traj


def evaluate_model(task, model):
    """
    给定 task 和 已训练好的 model，
    重新放一遍板，返回 (hpwl, slw, traj)
    """
    env = PCBEnv(task)
    env.reset()
    seq = chip_oriented_sequence(task)
    traj, _ = trajectory_generation(env, model, seq)# 用现有的 trajectory_generation 来放板
    wl, slw = env.score()# 此时 env 里已经是最后的布局状态，直接 score 一下
    return wl, slw, traj

def auto_tune_lam(task,
                  lam1=1000.0,
                  lam2_candidates=(0.0, 0.05, 0.1, 0.2, 0.3),
                  iters=3,
                  traj_per_iter=50,
                  batch_size=8,
                  lr=1e-3):
    """
    简单的 lam2 调参小助手：
    - 固定 lam1
    - 在 lam2_candidates 中依次尝试不同的 lam2
    - 每个 lam2 跑一小段训练，然后评估 HPWL / SLW
    - 选出 HPWL 最小的 lam2；如果多个 HPWL 接近，再偏向 SLW 较大的模型
    """
    best_model = None
    best_cfg = None
    best_hpwl = float("inf")
    best_slw = -1

    for lam2 in lam2_candidates:
        print(f"\n[LamAgent] Trying lam1={lam1}, lam2={lam2:.3f}")

        # 用当前的 lam1/lam2 跑一小段 online_finetune
        model, _ = online_finetune(task,
                                   iters=iters,
                                   traj_per_iter=traj_per_iter,
                                   batch_size=batch_size,
                                   lr=lr,
                                   lam1=lam1,
                                   lam2=lam2)

        # 评估这个模型的 HPWL / SLW
        hpwl, slw, _ = evaluate_model(task, model)
        print(f"[LamAgent] lam2={lam2:.3f} => HPWL={hpwl:.1f}, SLW={slw}")

        # 选优规则：
        # 1) HPWL 更小的优先
        # 2) 如果 HPWL 几乎一样（差距很小），SLW 更大的优先
        better = False
        if hpwl < best_hpwl - 1e-6:
            better = True
        elif abs(hpwl - best_hpwl) <= 1e-6 and slw > best_slw:
            better = True

        if better:
            best_hpwl = hpwl
            best_slw = slw
            best_model = model
            best_cfg = (lam1, lam2)

    print(f"\n[LamAgent] Best config: lam1={best_cfg[0]}, lam2={best_cfg[1]:.3f}, "
          f"HPWL={best_hpwl:.1f}, SLW={best_slw}")
    return best_model, best_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="tasks/sample_task.json")  # 不再 required
    ap.add_argument("--iters", type=int, default=6) #iters 微调轮数
    ap.add_argument("--traj_per_iter", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--expert_task", type=str, default="expert_traj/expert1_task.json",
                        help="专家板子的 task.json（如果为空则不做专家预训练）")
    ap.add_argument("--expert_layout", type=str, default="expert_traj/expert1_layout.json",
                        help="专家板子的 layout.json")
    ap.add_argument("--bc_epochs", type=int, default=20,
                        help="行为克隆预训练轮数（0 表示跳过）")
    ap.add_argument("--bc_lr", type=float, default=1e-3,
                        help="行为克隆预训练学习率")

    ap.add_argument("--only_bc", action="store_true",
                    help="只做专家行为克隆，不做 RL 微调")

    ap.add_argument(
        "--auto_tune",
        action="store_true",
        # default=True,
        help="启用 Lam 参数自动调节（不加这个参数则走普通 online_finetune）,默认开启",
    )
    args = ap.parse_args()
    model = None

    # 如果指定了 expert_task / layout + bc_epochs>0，先做行为克隆
    task = None #为了在 if/else 两个分支里都赋值 task，先设成 None
    if args.expert_task is not None and args.expert_layout is not None and args.bc_epochs > 0:
        print("[main] pretrain on expert...")
        model, task = pretrain_on_expert(
            expert_task_path=args.expert_task,
            expert_layout_path=args.expert_layout,
            epochs=args.bc_epochs,
            lr=args.bc_lr,
        )
    else:
        # 否则就用普通的 task
        task = json.load(open(args.task, "r", encoding="utf-8"))
        model = TinyDT(task["grid_N"])

        # 2) 如果只想看 BC 效果，就直接 rollout 一条轨迹
    if args.only_bc:
        env = PCBEnv(task)
        env.reset()
        seq = chip_oriented_sequence(task)
        traj, score = trajectory_generation(env, model, seq)
        wl, slw = env.score()
        print(f"[BC-Only] HPWL={wl:.1f}, SLW={slw}, score={score:.4f}")
        torch.save(model.state_dict(), "ckpt_bc.pt")
        print("[train] saved ckpt_bc.pt; final traj len=", len(traj))
        return

        # 用你原来的 online_finetune 做自博弈微调（RL + 优先回放）
    model, traj = online_finetune(
        task,
        iters=args.iters,
        traj_per_iter=args.traj_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        model=model,  # 记得把已经预训练过的 model 传进去（如果你在线里支持）；不做专家预训练时传None
    )
    #用训练好的模型再放一遍板 → 打印最终 HPWL/SLW + score，并保存 ckpt
    env = PCBEnv(task)
    env.reset()
    seq = chip_oriented_sequence(task)
    traj, score = trajectory_generation(env, model, seq)
    wl, slw = env.score()
    print(f"[Final] HPWL={wl:.1f}, SLW={slw}, score={score:.4f}", flush=True)
    torch.save(model.state_dict(), "ckpt.pt")
    print("[train] saved ckpt.pt; final traj len=", len(traj))

if __name__ == "__main__":
    main()



#拿着这个已经学会专家行为的模型，跑 online_finetune：自己再在同一块板子上放若干局；收集好的轨迹，放进优先回放池；用轨迹质量作为权重，对自己的行为做加权行为克隆；迭代 6 轮。