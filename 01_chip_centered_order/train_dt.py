"""Minimal Decision Transformer supervised training loop using expert trajectories.

Usage: Run to perform a small sanity training pass over expert data, preserving
raw layout positions (placement_mode='layout'). Failing trajectories are skipped.
"""
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from datasets.expert_dataset import ExpertTrajectoryDataset
from DT.models.dt_model import DecisionTransformer


def collect_samples(dataset: ExpertTrajectoryDataset, max_samples: int) -> List:
    samples = []
    for idx in range(len(dataset)):
        if len(samples) >= max_samples:
            break
        try:
            samples.append(dataset[idx])
        except Exception as e:
            print(f"[skip] index={idx} error={e}")
    return samples


def batchify(samples: List, batch_size: int):
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def compute_loss(batch, model: DecisionTransformer, device: torch.device):
    states = batch["states"].to(device)            # (B, T, 4, N, N)
    actions = batch["actions"].to(device)          # (B, T, 3)
    returns_to_go = batch["returns_to_go"].to(device)
    timesteps = batch["timesteps"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Forward
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=None,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    x_logits, y_logits, rot_logits = action_preds  # each (B, T, ...)

    # Targets
    x_targets = actions[:, :, 0]
    y_targets = actions[:, :, 1]
    rot_targets = actions[:, :, 2]

    # Mask flattening
    mask = attention_mask.bool()  # (B, T)
    # Flatten valid positions
    def masked_ce(logits: torch.Tensor, targets: torch.Tensor):
        # logits shape (B, T, C); targets (B, T)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)[mask.view(-1)]
        targets_flat = targets.view(B * T)[mask.view(-1)]
        return F.cross_entropy(logits_flat, targets_flat)

    loss_x = masked_ce(x_logits, x_targets)
    loss_y = masked_ce(y_logits, y_targets)
    loss_rot = masked_ce(rot_logits, rot_targets)

    # Optional: predictive next position regression (state_preds) & return prediction
    # Use masked MSE for returns as auxiliary
    returns_target = returns_to_go  # already (B, T, 1)
    returns_pred = return_preds  # (B, T, 1)
    returns_mse = F.mse_loss(returns_pred[mask], returns_target[mask])

    loss = loss_x + loss_y + loss_rot + 0.1 * returns_mse
    return loss, {
        "loss": loss.item(),
        "loss_x": loss_x.item(),
        "loss_y": loss_y.item(),
        "loss_rot": loss_rot.item(),
        "returns_mse": returns_mse.item(),
    }


def _draw_board(component_boxes: List[Tuple[int,int,int,int,int]], grid_size: int, title: str, out_path: Path):
    """component_boxes: list of (x, y, w, h, rot)."""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(title)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for (x,y,w,h,rot) in component_boxes:
        rect = Rectangle((x,y), w, h, linewidth=0.6, edgecolor='C0', facecolor='none')
        ax.add_patch(rect)
        ax.text(x+ w/2, y+ h/2, f"r{rot}", fontsize=6, ha='center', va='center')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="dta/expert_traj (2)/expert_traj", help="Path to expert_traj folders")
    ap.add_argument("--grid-size", type=int, default=128)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--placement-mode", type=str, choices=["layout", "heuristic"], default="layout")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)

    dataset = ExpertTrajectoryDataset(
        root_dir=data_root,
        split="test",
        grid_size=args.grid_size,
        margin=args.margin,
        placement_mode=args.placement_mode,
        preload=False,
    )

    # Collect a small subset for quick smoke test training
    samples = collect_samples(dataset, max_samples=args.max_samples)
    print(f"Collected {len(samples)} usable expert trajectories for training subset")

    model = DecisionTransformer(grid_size=args.grid_size, hidden_size=256)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    epochs = args.epochs
    batch_size = args.batch_size

    loss_history = []
    reward_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_reward = 0.0
        reward_count = 0
        count = 0
        epoch_hpwl_sum = 0.0
        epoch_slw_sum = 0.0
        epoch_nslw_sum = 0.0
        epoch_score_sum = 0.0
        epoch_metric_count = 0
        for batch_samples in batchify(samples, batch_size):
            batch = ExpertTrajectoryDataset.collate_fn(batch_samples)
            optimizer.zero_grad()
            loss, metrics = compute_loss(batch, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            # 奖励统计 (masked 平均)
            rewards = batch["rewards"].to(device)  # (B,T,1)
            mask = batch["attention_mask"].to(device).bool()  # (B,T)
            masked_rewards = rewards[mask]
            if masked_rewards.numel() > 0:
                total_reward += masked_rewards.mean().item()
                reward_count += 1

            # 可选：如果 batch 包含 hpwl/slw/nslw/score 则统计其 masked 平均
            if 'hpwl' in batch and batch['hpwl'] is not None:
                hpwl = batch['hpwl'].to(device)
                hpwl_masked = hpwl[mask]
                if hpwl_masked.numel() > 0:
                    epoch_hpwl_sum += hpwl_masked.mean().item()
                    epoch_metric_count += 1
            if 'slw' in batch and batch['slw'] is not None:
                slw = batch['slw'].to(device)
                slw_masked = slw[mask]
                if slw_masked.numel() > 0:
                    epoch_slw_sum += slw_masked.mean().item()
            if 'nslw' in batch and batch['nslw'] is not None:
                nslw = batch['nslw'].to(device)
                nslw_masked = nslw[mask]
                if nslw_masked.numel() > 0:
                    epoch_nslw_sum += nslw_masked.mean().item()
            if 'score' in batch and batch['score'] is not None:
                score = batch['score'].to(device)
                score_masked = score[mask]
                if score_masked.numel() > 0:
                    epoch_score_sum += score_masked.mean().item()

        avg_loss = total_loss / max(count, 1)
        avg_reward = total_reward / max(reward_count, 1) if reward_count else 0.0
        loss_history.append(avg_loss)
        reward_history.append(avg_reward)

        # Compose log message with optional metrics
        msg = f"Epoch {epoch+1}: avg_loss={avg_loss:.4f} avg_reward={avg_reward:.4f}"
        if epoch_metric_count > 0:
            msg += f" avg_hpwl={epoch_hpwl_sum/epoch_metric_count:.4f}"
        if epoch_slw_sum > 0:
            msg += f" avg_slw={epoch_slw_sum/epoch_metric_count:.4f}"
        if epoch_nslw_sum > 0:
            msg += f" avg_nslw={epoch_nslw_sum/epoch_metric_count:.4f}"
        if epoch_score_sum > 0:
            msg += f" avg_score={epoch_score_sum/epoch_metric_count:.4f}"
        print(msg)

    # Save a checkpoint
    ckpt_path = Path("output") / "dt_smoke.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"grid_size": args.grid_size, "hidden_size": 256}}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 训练曲线绘制
    fig, ax = plt.subplots(2,1, figsize=(6,6))
    ax[0].plot(loss_history, marker='o')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(reward_history, marker='o', color='C2')
    ax[1].set_title('Reward Curve (avg per epoch)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Reward')
    fig.tight_layout()
    curve_path = Path('output') / 'training_curves.png'
    fig.savefig(curve_path)
    plt.close(fig)
    print(f"Saved curves to {curve_path}")

    # 可视化一个样本的原始 vs 模型预测布图 (选第一条成功样本)
    if samples:
        sample = samples[0]
        processor = dataset.get_processor(0)
        comp_id_to_size = {c['comp_id']: c['size'] for c in processor.component_list}
        original_boxes = []
        for a, comp in zip(sample.actions.tolist(), getattr(processor.build_env(), 'placement_sequence', processor.component_list)):
            x,y,r = a
            size = comp_id_to_size.get(comp['comp_id'], [1,1])
            w,h = size
            if r % 2 == 1:
                w,h = h,w
            original_boxes.append((x,y,w,h,r))

        seq_states = sample.states.clone()
        seq_actions = sample.actions.clone()
        seq_returns = sample.returns_to_go.clone()
        seq_timesteps = sample.timesteps.clone()
        try:
            last_action = model.get_action(
                states=seq_states.to(device),
                actions=seq_actions.to(device),
                rewards=None,
                returns_to_go=seq_returns.to(device),
                timesteps=seq_timesteps.to(device),
                deterministic=True
            )
        except Exception:
            last_action = (0,0,0)

        median_size = [int(torch.median(torch.tensor([s[0] for s in comp_id_to_size.values()]))) if comp_id_to_size else 1,
                       int(torch.median(torch.tensor([s[1] for s in comp_id_to_size.values()]))) if comp_id_to_size else 1]
        pw,ph = median_size
        r = int(last_action[2]) if hasattr(last_action, '__len__') else 0
        if r % 2 == 1:
            pw,ph = ph,pw
        predicted_boxes = original_boxes[:-1] + [(int(last_action[0]), int(last_action[1]), pw, ph, r)]

        board_orig_path = Path('output') / 'board_original.png'
        board_pred_path = Path('output') / 'board_predicted.png'
        _draw_board(original_boxes, processor.grid_size, 'Original (Expert/Replay)', board_orig_path)
        _draw_board(predicted_boxes, processor.grid_size, 'Predicted (Last Action Replaced)', board_pred_path)
        print(f"Saved board visuals to {board_orig_path} and {board_pred_path}")


if __name__ == "main" or __name__ == "__main__":  # handle both conventions
    main()
