import argparse, json, torch
from env import PCBEnv
from seq import chip_oriented_sequence
from model import TinyDT
from replay import PrioritizedReplay
import torch
import numpy as np

# 将 (gx, gy) 转为实坐标；与 env.py 的口径保持一致（使用 boundary[0]/[2]）
def grid_to_xy(gx, gy, env):
    xmin, ymin = env.boundary[0]
    xmax, ymax = env.boundary[2]
    W = xmax - xmin; H = ymax - ymin
    return xmin + (gx + 0.5) * (W / env.N), ymin + (gy + 0.5) * (H / env.N)

# 统一的打分函数：根据 HPWL(wl) 和 SLW(slw) 计算 score
def compute_score(wl, slw, lam1=1000.0, lam2=0.1):
    """
    wl   : HPWL（越小越好）
    slw  : 可直线布线的二端网数量（越大越好）
    lam1 : 线长的权重
    lam2 : SLW 的权重

    返回值 score 越大代表轨迹越好。
    这里保持你原来的公式不变：lam1*(1/wl) + lam2*slw
    """
    return lam1 * (1.0 / max(wl, 1e-6)) + lam2 * slw

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
def traj_loss(model, traj_steps, normalize_w= True):# 计算轨迹的负对数概率损失
    logps = []
    for tokens, (gx,gy,rot) in traj_steps:
        logits_by_rot = model(tokens)               # {rot: N*N logits}
        idx = gy * model.N + gx                     # 展平索引
        logits = logits_by_rot[rot]                 # 这一步对应的朝向下，对所有格点的logits
        logp = torch.log_softmax(logits, dim=0)[idx] # 该动作的 log 概率
        logps.append(logp)
    return -torch.stack(logps).sum() #把每步的 log 概率相加，取负号作为负对数似然 (NLL)，最小化它等价于让模型更倾向于复现这条轨迹里的动作（即行为克隆/最大似然）
#task 是 PCB 布局任务的描述字典； iters 是微调轮数； traj_per_iter 是每轮收集的合法轨迹数； batch_size 是每轮训练批次大小； lr 是学习率
def online_finetune(task, iters=10, traj_per_iter=100, batch_size=8, lr=1e-3, lam1=1000.0, lam2=0.1):
    env = PCBEnv(task)
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
            for name in seq:
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
                collected.append((steps, score))
        total_collected += len(collected)
        print(f"[iter {it}] collected={len(collected)}  total={total_collected}  buffer={len(buffer)}")

        # ---- 训练 ----
        # 从回放池取一批（这里示例简单用最近收集的，可以修改成从buffer按优先级采样）
        batch_items = buffer.topk(batch_size, with_scores=True)  # [(payload, score), ...]
        # payload 是你 add 时塞进去的 (steps, traj)
        steps_list = []
        scores = []
        for (payload, sc) in batch_items:
            steps, _traj = payload
            steps_list.append(steps)
            scores.append(float(sc))

        scores = np.array(scores, dtype=np.float32)

        # 归一化权重（与之前一致）
        w = (scores - scores.mean()) / (scores.std() + 1e-6)
        w = np.clip(w, -2.0, 2.0)
        w = np.exp(w)
        w = w / (w.mean() + 1e-6)

        opt.zero_grad()
        loss = 0.0
        for steps, wi in zip(steps_list, w):
            loss = loss + (traj_loss(model, steps) * float(wi))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

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
    # 用现有的 trajectory_generation 来放板
    traj, _ = trajectory_generation(env, model, seq)
    # 此时 env 里已经是最后的布局状态，直接 score 一下
    wl, slw = env.score()
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
    ap.add_argument(
        "--auto_tune",
        action="store_true",
        default=True,
        help="启用 Lam 参数自动调节（不加这个参数则走普通 online_finetune）,默认开启",
    )
    args = ap.parse_args()
    task = json.load(open(args.task, "r"))

    if args.auto_tune:
        # 用 LamAgent 自动选择 lam2
        best_model, best_cfg = auto_tune_lam(task,
                                             lam1=1000.0,
                                             lam2_candidates=(0.0, 0.05, 0.1, 0.2, 0.3),
                                             iters=3,
                                             traj_per_iter=50,
                                             batch_size=8,
                                             lr=1e-3)
        print("Best lam config:", best_cfg)
        model = best_model
        # 然后你可以用 best_model 再跑一次 trajectory_generation 画最终图
    else:
        # 原来的训练逻辑
        model, traj = online_finetune(task, iters=args.iters, traj_per_iter=args.traj_per_iter)

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
