import numpy as np
import torch
import torch.nn as nn

import transformers

from .base_model import TrajectoryModel
from .gpt2_model import GPT2Model
from .pcb_state_encoder import PCBStateEncoder, PCBStateEncoderViT


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    Modified for PCB layout: states are represented as 4-channel mask images instead of vectors
    """

    def __init__(
        self,
        grid_size=32,      # PCB网格大小 (N x N)
        hidden_size=128,
        max_length=None,
        max_ep_len=4096,
        use_vit=False,     # 是否使用ViT编码器，否则使用CNN
        hi_dim=11,
        **kwargs
    ):
        # PCB布局动作空间固定为 (x, y, rotation)
        act_dim = 3  # 固定：x坐标, y坐标, 旋转角度
        super().__init__(grid_size, act_dim, max_length=max_length)

        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.hi_dim = hi_dim
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_head=4,      # 确保hidden_size能被n_head整除
            n_positions=1024,
        )
        # 允许kwargs覆盖默认配置
        for key, value in kwargs.items():
            setattr(config, key, value)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # PCB状态编码器 (替换原来的线性嵌入)
        if use_vit:
            self.embed_state = PCBStateEncoderViT(grid_size=grid_size, hidden_size=hidden_size)
        else:
            self.embed_state = PCBStateEncoder(grid_size=grid_size, hidden_size=hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_hi = torch.nn.Linear(hi_dim, hidden_size)

        # PCB专用动作嵌入：离散动作空间 (x, y, rotation)
        self.embed_x = nn.Embedding(self.grid_size, hidden_size // 4)  # x坐标嵌入
        self.embed_y = nn.Embedding(self.grid_size, hidden_size // 4)  # y坐标嵌入
        self.embed_rotation = nn.Embedding(4, hidden_size // 2)        # 旋转嵌入 (更大的容量)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        # 预测状态现在是预测下一个组件的位置 (x, y坐标)
        self.predict_state = torch.nn.Linear(hidden_size, 2)  # 预测 (x, y) 位置坐标
        # PCB专用动作预测头：结构化离散动作空间 (x, y, rotation)
        # X坐标分类器：预测x坐标 (0到grid_size-1)
        self.predict_x = nn.Linear(hidden_size, self.grid_size)
        # Y坐标分类器：预测y坐标 (0到grid_size-1)
        self.predict_y = nn.Linear(hidden_size, self.grid_size)
        # 旋转分类器：预测旋转角度 (0°, 90°, 180°, 270°)
        self.predict_rotation = nn.Linear(hidden_size, 4)
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, hpwl_threshold=None, bonus_value: float = -0.01, hi=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(states.device)

        # embed each modality with a different head
        # states shape: (batch_size, seq_length, 4, N, N)
        # 需要对每个时间步的state分别编码
        batch_size, seq_length = states.shape[0], states.shape[1]
        state_embeddings_list = []
        for t in range(seq_length):
            # 取出第t个时间步的state: (batch_size, 4, N, N)
            state_t = states[:, t]  # (batch_size, 3, N, N)
            # 编码为特征向量: (batch_size, hidden_size)
            state_embed_t = self.embed_state(state_t)
            state_embeddings_list.append(state_embed_t)

        # 拼接所有时间步的嵌入: (batch_size, seq_length, hidden_size)
        state_embeddings = torch.stack(state_embeddings_list, dim=1)
        
        # PCB专用动作嵌入：将离散动作(x,y,rotation)嵌入到连续空间
        # actions shape: (batch_size, seq_length, 3) - [x, y, rotation]
        x_actions = actions[:, :, 0].long()  # x坐标索引
        y_actions = actions[:, :, 1].long()  # y坐标索引
        rot_actions = actions[:, :, 2].long()  # 旋转索引
        
        x_embeddings = self.embed_x(x_actions)      # (batch, seq, hidden//3)
        y_embeddings = self.embed_y(y_actions)      # (batch, seq, hidden//3)
        rot_embeddings = self.embed_rotation(rot_actions)  # (batch, seq, hidden//3)
        
        # 拼接三个嵌入向量
        action_embeddings = torch.cat([x_embeddings, y_embeddings, rot_embeddings], dim=-1)
        
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        if hi is not None:
            if hi.dim() == 1:
                hi = hi.unsqueeze(0)
            hi = hi.to(states.device)
            hi_embed = self.embed_hi(hi)  # (B, hidden)
            hi_expand = hi_embed.unsqueeze(1).expand(-1, seq_length, -1)
            state_embeddings = state_embeddings + hi_expand
            action_embeddings = action_embeddings + hi_expand
            returns_embeddings = returns_embeddings + hi_expand

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length).to(states.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next position (x,y) given state and action
        # PCB专用：预测结构化离散动作
        x_logits = self.predict_x(x[:,1])      # predict x coordinate logits
        y_logits = self.predict_y(x[:,1])      # predict y coordinate logits
        rot_logits = self.predict_rotation(x[:,1])  # predict rotation logits (4 classes)

        # 应用Position Mask确保只预测合法动作，并根据论文注入bonus与HPWL阈值抑制
        x_logits, y_logits = self._apply_position_and_guides(
            x_logits, y_logits, rot_logits, states, hpwl_threshold, bonus_value
        )

        action_preds = (x_logits, y_logits, rot_logits)  # 返回带掩码的logits元组

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, deterministic=False, temperature=1.0, hi=None, **kwargs):
        # we don't care about the past rewards in this model

        # states shape: (seq_len, 4, N, N) - image format
        device = next(self.parameters()).device
        states = states.unsqueeze(0).to(device)  # add batch dimension & move to device
        actions = actions.reshape(1, -1, 3).to(device)  # (seq_len,3) -> (1, seq_len, 3)
        returns_to_go = returns_to_go.reshape(1, -1, 1).to(device)
        timesteps = timesteps.reshape(1, -1).to(device)
        if hi is not None:
            hi_tensor = hi.to(device)
            if hi_tensor.dim() == 1:
                hi_tensor = hi_tensor.unsqueeze(0)
        else:
            hi_tensor = None

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])]).to(device)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            # pad states (images)
            pad_length = self.max_length - states.shape[1]
            if pad_length > 0:
                pad_states = torch.zeros((1, pad_length, 4, self.grid_size, self.grid_size), device=device)
                states = torch.cat([pad_states, states], dim=1)

            # pad actions - 现在是离散动作 [x, y, rotation]
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], 3),
                             device=device), actions],
                dim=1).to(dtype=torch.long)  # 离散动作用long类型

            # pad returns_to_go
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=device), returns_to_go],
                dim=1).to(dtype=torch.float32)

            # pad timesteps
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, hi=hi_tensor, **kwargs
        )

        # 从logits中提取动作
        x_logits, y_logits, rot_logits = action_preds

        if deterministic:
            # 确定性：选择最可能的动作
            x_pred = torch.argmax(x_logits[0, -1])
            y_pred = torch.argmax(y_logits[0, -1])
            rot_pred = torch.argmax(rot_logits[0, -1])
        else:
            # 随机采样：使用温度控制的随机性
            x_probs = torch.softmax(x_logits[0, -1] / temperature, dim=-1)
            y_probs = torch.softmax(y_logits[0, -1] / temperature, dim=-1)
            rot_probs = torch.softmax(rot_logits[0, -1] / temperature, dim=-1)

            x_pred = torch.multinomial(x_probs, 1).squeeze()
            y_pred = torch.multinomial(y_probs, 1).squeeze()
            rot_pred = torch.multinomial(rot_probs, 1).squeeze()

        # 返回离散动作元组 (x, y, rotation)
        return (x_pred.item(), y_pred.item(), rot_pred.item())

    def _apply_position_and_guides(self, x_logits, y_logits, rot_logits, states, hpwl_threshold=None, bonus_value: float = -0.01):
        """
        合法化与软约束引导（论文对齐）：
        - Position Mask：抑制非法位置（统一语义：0=合法，1=非法）
        - Bonus：对满足对齐/最小间距的合法坐标注入小负值，提升被选概率
        - Wire Mask：对超过HPWL阈值的坐标进一步抑制（若提供阈值）
        """
        batch_size, seq_len = x_logits.shape[0], x_logits.shape[1]
        grid_size = self.grid_size
        device = x_logits.device
        # 约定：states 的通道包括 [view, position_mask, wire_mask, bonus_mask]
        # position_mask: 0=合法，1=非法
        # wire_mask: HPWL增量的实值矩阵
        # bonus_mask: 合法且满足对齐/最小间距的位置为1，其它为0
        position_masks = states[:, :, 1].clamp(0, 1)  # (batch, seq, N, N)
        wire_masks = states[:, :, 2]                  # (batch, seq, N, N)
        bonus_masks = states[:, :, 3]                 # (batch, seq, N, N)

        # 为每个batch和sequence位置应用掩码
        masked_x_logits = x_logits.clone()
        masked_y_logits = y_logits.clone()

        for b in range(batch_size):
            for t in range(seq_len):
                # 获取当前时间步的Position/Wire/Bonus Mask
                pos_mask = position_masks[b, t]  # (N, N) 二值掩码，0=合法，1=非法
                wire_mask = wire_masks[b, t]     # (N, N) HPWL增量
                bonus_mask = bonus_masks[b, t]   # (N, N) 1=满足对齐/最小间距且合法
                
                # 合法性：pos_mask 为 0 的位置是合法
                legal_rows = (pos_mask == 0).float().sum(dim=1)  # 每个x是否有合法y
                legal_cols = (pos_mask == 0).float().sum(dim=0)  # 每个y是否有合法x
                x_validity = (legal_rows > 0).float()
                y_validity = (legal_cols > 0).float()

                # 1) 合法化：非法坐标 logits 设为极小值
                masked_x_logits[b, t] = torch.where(
                    x_validity > 0, 
                    x_logits[b, t], 
                    torch.full_like(x_logits[b, t], -1e9)
                )
                masked_y_logits[b, t] = torch.where(
                    y_validity > 0,
                    y_logits[b, t], 
                    torch.full_like(y_logits[b, t], -1e9)
                )

                # 2) Bonus 注入：对 bonus_mask==1 的合法坐标在对应 logits 上加小负值
                # 将二维 bonus 聚合到 x/y 维度上
                bonus_rows = (bonus_mask > 0).float().sum(dim=1)  # (N,)
                bonus_cols = (bonus_mask > 0).float().sum(dim=0)  # (N,)
                masked_x_logits[b, t] = masked_x_logits[b, t] + (bonus_rows > 0).float() * bonus_value
                masked_y_logits[b, t] = masked_y_logits[b, t] + (bonus_cols > 0).float() * bonus_value

                # 3) Wire 阈值抑制：超过阈值的坐标进一步降低概率（若提供阈值）
                if hpwl_threshold is not None:
                    wire_rows = (wire_mask.mean(dim=1) > hpwl_threshold).float()
                    wire_cols = (wire_mask.mean(dim=0) > hpwl_threshold).float()
                    masked_x_logits[b, t] = torch.where(
                        wire_rows > 0,
                        masked_x_logits[b, t] + torch.full_like(masked_x_logits[b, t], -5.0),  # 适度抑制
                        masked_x_logits[b, t]
                    )
                    masked_y_logits[b, t] = torch.where(
                        wire_cols > 0,
                        masked_y_logits[b, t] + torch.full_like(masked_y_logits[b, t], -5.0),
                        masked_y_logits[b, t]
                    )
        
        return masked_x_logits, masked_y_logits