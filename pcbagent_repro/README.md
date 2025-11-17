PCBAgent: An Agent-based Framework for High-Density PCB Placement：
- RL Agent：基于 Decision Transformer 的策略（简化实现）。
- 约束合法化：不重叠、边界、最小间距的 位置掩码（position mask）。
- 软约束优化：HPWL 引导的 wire mask + pad 对齐/最小间距奖励。
- 轨迹生成（Algorithm 1）与在线微调（Algorithm 2）的训练循环雏形。
- 指标：HPWL、NSLW（按论文定义给出可运行的近似检测器）。
- 组件摆放顺序：Chip-oriented 排序策略雏形。

## 目录
- `tasks/sample_task.json`：最小任务示例。
- `env.py`：PCB 环境，掩码构造（view/boundary/spacing/position/wire），HPWL/NSLW 计算。
- `seq.py`：Chip-oriented 摆放顺序。
- `model.py`：简化 Decision Transformer（DT）与策略采样接口。
- `replay.py`：优先级回放（Prioritized Replay）。
- `train.py`：Algorithm 2（在线微调）。
- `infer.py`：Algorithm 1（轨迹生成）。
- `eval_layout.py`：评估指标（HPWL、NSLW、合法性三元组）。
- `tools`：工具。
    - `refine_layout.py`：优化布置。
    - `drwa_layout.py`：画线。
    - `sweep.py`：扫描，修改三个参数，绘制版图。


## 与论文的对应关系

- 约束与目标函数、SLW 定义（已按文中给出实现/近似）。
- 掩码：view/position/wire，包含 b 奖励与 HPWL 阈值裁剪（已实现）。
- 旋转角：支持 {0,90,180,270}（已实现简化）。
- 摆放顺序：芯片优先，顺时针等（雏形）。
- DT：采用 Transformer 编码状态 + 因果掩码（简化），支持在线微调（已实现简化）。
- LLM agent：以命令行参数模拟“偏好/反馈”对权重与奖励的调整（雏形）。


