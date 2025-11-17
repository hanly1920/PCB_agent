#给定某个器件在四种朝向下的三张NXN栅格图，预测该器件的放置位置和朝向
import torch
import torch.nn as nn

#TinyDT 把每个器件“四种朝向 × 3 通道的 N×N 图”打平成向量，做一个 MLP+rot embedding，输出 N² 个 logits
class TinyDT(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.N = N  # 栅格边长
        in_dim = 3 * N * N    # 3 个通道：position / wire / bonus
        hid = 256
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.rot_emb = nn.Embedding(4, 32)  # 0/90/180/270
        self.head = nn.Linear(hid + 32, N * N)

    def forward(self, tokens_dict, mask_illegal: bool = True):
        """
        mask_illegal=True: 用 position 通道把非法格减 1e9（正常 RL / 采样）
        mask_illegal=False: 不做掩码（用于专家行为克隆，允许“专家动作”出现在 env 认为非法的格子）
        """
        device = next(self.parameters()).device
        logits_by_rot = {}
        for idx, rot in enumerate([0, 90, 180, 270]):
            t = tokens_dict[rot]
            # 转 tensor，并展平
            pos   = torch.as_tensor(t["position"], dtype=torch.float32, device=device).reshape(-1)
            wire = torch.zeros_like(pos)
            #wire  = torch.as_tensor(t["wire"],     dtype=torch.float32, device=device).reshape(-1)
            bonus = torch.as_tensor(t["bonus"],    dtype=torch.float32, device=device).reshape(-1)

            x = torch.cat([pos, wire, bonus], dim=0)                 # [3*N*N]
            h = self.encoder(x)                                       # [hid]
            rot_idx = torch.tensor([idx], dtype=torch.long, device=device)
            h = torch.cat([h, self.rot_emb(rot_idx).squeeze(0)], dim=0)  # [hid+32]
            logits = self.head(h)                                     # [N*N]

            if mask_illegal:
                # 只在 RL / 采样阶段做非法格惩罚
                logits = logits - pos * 1e9

            logits_by_rot[rot] = logits
        return logits_by_rot

    def sample_action(self,
                      tokens_dict, #你的环境为当前器件在各个朝向下准备的 3 个 N×N 通道：position(1=非法,0=合法)、wire(HPWL代价/增量)、bonus(对齐奖励)
                      temperature: float = 0.2, # 采样温度, 越低越贪婪 0.2很保守
                      allowed_rots=(0, 90, 180, 270), #允许的朝向集合，可用(0,90,270) 实现“3N²”
                      topk: int = 128, #只在联合 logits的前 k 名里采样，防尾部坏解
                      wire_lambda: float = 0.3, # 惩罚强度，越大越不愿意选 wire 高的格点
                      wire_q: float = 0.95): #用合法格子的 wire 第 q 分位数做自适应归一化尺度，不同板子/步的数值范围会自动适配（0.9~0.99 常用）
        """
        在 (R×N×N) 联合空间抽样，带“自适应 wire 软惩罚 + top-k 限制”。
          - wire_lambda: 惩罚强度（0.2~0.8 之间调）
          - wire_q: 用合法格子的 wire 第 q 分位数做归一化（0.9~0.99 之间调）
          - topk: 只在 top-k 里采样，避免尾部分布垃圾解
        """
        device = next(self.parameters()).device #把模型权重所在device（CPU/GPU）取出来，避免跨设备报错
        logits_by_rot = self(tokens_dict)  # 前向计算，得到各朝向的 logits
        N = self.N
        rots_seq = tuple(allowed_rots)
        flat_logits_list = [] #预备一个列表，待会把所有朝向的logits叠起来

        for r in rots_seq:
            logits_r = logits_by_rot[r].reshape(-1).to(device)  # [N*N]
            pos = torch.as_tensor(tokens_dict[r]["position"], dtype=torch.float32, device=device).reshape(-1)  #pos 是硬掩码（1=非法）
            wire = torch.as_tensor(tokens_dict[r]["wire"], dtype=torch.float32, device=device).reshape(-1) # wire 是代价

            legal = (pos < 0.5) #合法格掩码，position=0 表示合法，position=1 表示非法，所以 pos<0.5 就是合法格
            # 自适应归一化：用合法位置的 wire 第 wire_q 分位数做尺度，不同器件/步的 wire 量级会变
            if legal.any(): #确保有合法格，否则量化会报错，直接设 scale=1.0，避免除0
                scale = torch.quantile(wire[legal], torch.tensor(wire_q, device=device))
                scale = torch.clamp(scale, min=1e-6)
            else:
                scale = torch.tensor(1.0, device=device)

            penalty = wire / scale #规范化wire
            penalty = torch.clamp(penalty, 0.0, 10.0)  # 防止极端值
            penalty[~legal] = 0.0 #非法格惩罚设为0，避免影响合法格的logit
            # 软惩罚：减小高 wire 的 logit，但不直接禁用
            logits_r = logits_r - wire_lambda * penalty
            # 仍然对非法格硬禁：position=1 的全部置 -inf
            logits_r[~legal] = -1e9
            flat_logits_list.append(logits_r)

        flat_logits = torch.cat(flat_logits_list, dim=0) / max(temperature, 1e-6) # 联合 R*N*N 维度，并除以温度，温度越小，softmax 越尖锐，越接近贪心

        # top-k 采样，既保留探索又避免抽到尾部垃圾——保守探索
        if topk is not None and topk > 0 and topk < flat_logits.numel():#既保留探索（不是直接 argmax），又能避免抽到尾部概率极小但梯度数值混乱的位置
            vals, inds = torch.topk(flat_logits, k=topk)
            probs = torch.softmax(vals, dim=0)
            pick = torch.multinomial(probs, 1).item()
            idx = inds[pick].item()
        else: #否则在全体[R*N*N] 上采样
            probs = torch.softmax(flat_logits, dim=0)
            idx = torch.multinomial(probs, 1).item()

        #反解 idx 到 (gx, gy, rot)
        R = len(rots_seq)
        rot_idx = idx // (N * N) #把联合索引 idx 拆回“第几个朝向 + 第几个格子”
        cell_idx = idx % (N * N)
        gy, gx = divmod(cell_idx, N) #divmod(cell_idx, N) 把 [0..N*N-1] 映回网格坐标 (gy,gx)
        rot = rots_seq[rot_idx] #根据 rot_idx 取出实际朝向值（0/90/180/270）
        return int(gx), int(gy), int(rot)

