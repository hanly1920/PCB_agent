from __future__ import annotations

import torch

from pcbplace.train import BoardTrainingOutcome, _step_optimizer_if_trainable_batch


class _OneParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def test_all_failed_but_valid_prefix_batch_steps_optimizer():
    model = _OneParameterModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    outcomes = [
        BoardTrainingOutcome(
            loss=0.7,
            valid_steps=2,
            failed=True,
            failure_reason="illegal_step:overlap",
            suffix_complete=False,
            terminated=True,
        ),
        BoardTrainingOutcome(
            loss=0.4,
            valid_steps=1,
            failed=True,
            failure_reason="no_legal_action",
            suffix_complete=False,
            terminated=True,
        ),
    ]
    assert all(not outcome.curriculum_valid for outcome in outcomes)
    assert all(outcome.has_trainable_loss for outcome in outcomes)

    opt.zero_grad(set_to_none=True)
    loss = (model.weight - 3.0).pow(2).sum()
    loss.backward()
    before = model.weight.detach().clone()

    stepped = _step_optimizer_if_trainable_batch(
        model,
        opt,
        outcomes,
        max_grad_norm=999.0,
    )

    assert stepped is True
    assert not torch.equal(model.weight.detach(), before)


def test_all_failed_without_valid_prefix_skips_optimizer_step():
    model = _OneParameterModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    outcomes = [
        BoardTrainingOutcome(
            loss=None,
            valid_steps=0,
            failed=True,
            failure_reason="no_valid_loss",
            suffix_complete=False,
            terminated=True,
        )
    ]

    opt.zero_grad(set_to_none=True)
    loss = (model.weight - 3.0).pow(2).sum()
    loss.backward()
    before = model.weight.detach().clone()

    stepped = _step_optimizer_if_trainable_batch(model, opt, outcomes)

    assert stepped is False
    assert torch.equal(model.weight.detach(), before)
