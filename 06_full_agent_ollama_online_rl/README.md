# pcb_autoplace

`pcb_autoplace` 是一个 PCB 自动布局训练、强化微调、推理和 HTTP 服务工程。核心流程是把 KiCad PCB 转成结构化 JSON，用 CUDA 模型学习元件放置顺序与位置，再通过目标函数感知的推理策略生成布局结果。

本版本已经清理旧说明和缓存文件，并包含两处训练/推理一致性修复：

- 推理阶段默认从 checkpoint 的 `action_scoring` metadata 恢复 `infer_objective_alpha` 与 `infer_region_alpha`；只有在 CLI/API 显式传参时才覆盖 checkpoint 设置。
- replay rollout 统计会纳入首步失败的 episode，例如 `steps == 0` 的 `no_legal_action` 会进入 complete / illegal / no-legal / raw-objective 统计，但不会进入 policy-gradient 更新。

## 目录结构

```text
pcb_autoplace/
├── pcbplace/                  # 核心 Python 包：环境、模型、训练、推理、服务逻辑
├── scripts/                   # 命令行入口
├── tools/                     # 辅助检查和清理工具
├── tests/                     # 回归测试
├── data/
│   ├── train/                 # 训练 JSON 样本
│   ├── infer/                 # 推理 JSON 样本
│   ├── kicad/                 # 对应 KiCad PCB 原始训练/推理样本
│   └── audits/                # 转换中间文件与结构审计（不参与 glob）
├── README.md
└── requirements.txt
```

## 环境要求

建议使用 Python 3.10+。训练和默认推理路径面向 CUDA 环境，训练脚本会拒绝非 CUDA device。

安装依赖：

```bash
cd pcb_autoplace
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

如果你的 CUDA / PyTorch 版本有特定要求，优先按 PyTorch 官方方式安装匹配本机 CUDA 的 `torch`，再安装其它依赖：

```bash
pip install numpy visdom fastapi "uvicorn[standard]" pytest
```

## 快速检查

```bash
PYTHONPATH=. python -m compileall -q pcbplace scripts tools tests
PYTHONPATH=. pytest -q
python scripts/step3_train_masked.py --help
python scripts/step4_infer.py --help
```

## 数据准备

### 1. KiCad PCB 转 JSON

`step1_kicad_to_json.py` 支持处理一个 KiCad 根目录或单个 expert 目录：

```bash
python scripts/step1_kicad_to_json.py \
  --kicad_root path/to/kicad_root_or_expert_dir \
  --out_dir data/json \
  --grid_mm 1.0
```

可选参数：

```bash
--only_expert expert1
--interface_pad_mode numeric   # numeric / exclude / all
```

### 2. 生成结构化训练/推理 JSON

训练模式允许使用 expert/final layout 字段构造监督标签：

```bash
python scripts/generate_training_structure.py \
  --input data/json/board.json \
  --output data/train/board.structured.json \
  --mode train
```

推理模式会使用 leakage-safe 处理，不依赖 expert layout：

```bash
python scripts/generate_training_structure.py \
  --input data/json/board.json \
  --output data/test/board.structured.json \
  --mode infer
```

默认会在输出文件旁生成审计文件：

```text
board.structured.json.audit.json
```

## 训练

基础训练：

```bash
python scripts/step3_train_masked.py \
  --train_glob "data/train/*.json" \
  --save_path model.pt \
  --device cuda
```

常用控制参数：

```bash
--steps 0                    # <=0 表示自动训练到 board graduation
--warmup 1500
--batch_size 1
--max_tokens 128
--checkpoint_path ckpt.pt
--checkpoint_every_steps 1000
--resume
```

replay / 强化微调默认开启：

```bash
--replay_finetune
--replay_iters 20
--replay_rollouts_per_iter 16
--replay_update_steps 32
--replay_policy_gradient_coef 1.0
--replay_entropy_coef 0.01
--replay_pg_policy_mode shaped
```

建议保持 `--replay_pg_policy_mode shaped`，它和 rollout / inference 的 action scoring 一致。`pure_logits` 仅适合作为消融实验。

checkpoint 会保存模型结构、训练配置、sequence policy、layout objective 配置以及 action scoring metadata。

## 推理

```bash
python scripts/step4_infer.py \
  --test_glob "data/test/*.json" \
  --ckpt model.pt \
  --out_dir runs/infer \
  --device cuda
```

默认行为：

- 严格加载 checkpoint 权重；结构不匹配会报错。
- `--sequence_policy checkpoint` 会优先恢复训练时保存的 sequence policy。
- `--layout_preset checkpoint` 会优先恢复 checkpoint 里的布局目标权重。
- `--infer_objective_alpha` 和 `--infer_region_alpha` 不传时，会优先使用 checkpoint 的 `action_scoring` metadata；旧 checkpoint 没有该字段时回退到代码默认值。

显式覆盖 action scoring：

```bash
python scripts/step4_infer.py \
  --test_glob "data/test/*.json" \
  --ckpt model.pt \
  --out_dir runs/infer_override \
  --device cuda \
  --infer_objective_alpha 0.35 \
  --infer_region_alpha 0.10
```

启用 beam search：

```bash
python scripts/step4_infer.py \
  --test_glob "data/test/*.json" \
  --ckpt model.pt \
  --out_dir runs/infer_beam \
  --device cuda \
  --beam_width 4 \
  --beam_topk 32
```

旧 checkpoint 迁移时才建议使用：

```bash
--allow_partial_state_dict
```

## HTTP Reply 服务

启动服务：

```bash
python scripts/reply_server.py \
  --ckpt model.pt \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

健康检查：

```bash
curl http://localhost:8000/healthz
```

从结构化 JSON 文件推理：

```bash
curl -X POST http://localhost:8000/reply \
  -H "Content-Type: application/json" \
  -d '{
    "task_json_path": "data/test/example.json",
    "beam_width": 1,
    "beam_topk": 16,
    "return_metrics": true,
    "strict_state_dict": true
  }'
```

也可以传入 inline `task`：

```json
{
  "task": {
    "board": {},
    "components": [],
    "nets": []
  },
  "ckpt_path": "model.pt",
  "beam_width": 1,
  "return_metrics": true,
  "strict_state_dict": true
}
```

`task_json_path` 与 `task` 必须二选一。checkpoint 可通过请求里的 `ckpt_path`、服务启动参数 `--ckpt`，或环境变量 `PCBPLACE_CKPT` 提供。

## 评估和辅助工具

评估原始 / expert layout：

```bash
python scripts/eval_original_layouts.py \
  --glob "data/train/*.json" \
  --out_jsonl runs/original_eval.jsonl
```

评估推理输出的语义指标：

```bash
python scripts/eval_semantic_metrics.py \
  --task_glob "data/test/*.json" \
  --pred_dir runs/infer \
  --out runs/semantic_metrics.json
```

检查 runtime-visible semantic 信息：

```bash
python scripts/check_semantic_usage.py \
  --root . \
  --glob "data/train/*.json"
```

## 测试

```bash
PYTHONPATH=. pytest -q
```

当前回归测试覆盖：

- all-failed valid-prefix 仍会 optimizer step
- failed rollout objective penalty
- inference action-scoring checkpoint metadata 优先级
- runtime prior hint whitelist
- expert leakage guard
- review weight split

## 打包说明

发布包里不应包含以下生成物：

```text
__pycache__/
.pytest_cache/
*.pyc
*.pyo
```

重新打包示例：

```bash
cd ..
zip -r pcb_autoplace.zip pcb_autoplace \
  -x "*/__pycache__/*" "*/.pytest_cache/*" "*.pyc" "*.pyo"
```
