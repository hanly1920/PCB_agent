from pcbplace.train import _region_supervision_weights_from_row


def test_legacy_review_weight_only_gates_semantic_supervision():
    weights = _region_supervision_weights_from_row({"review_weight": 0.0})

    assert weights.semantic == 0.0
    assert weights.expert == 1.0
    assert weights.teacher == 1.0
    assert weights.metric == 1.0
    assert weights.prior == 1.0


def test_split_supervision_weights_accept_explicit_overrides_and_clamp():
    weights = _region_supervision_weights_from_row(
        {
            "review_weight": 0.2,
            "semantic_supervision_weight": 0.4,
            "expert_supervision_weight": 1.2,
            "teacher_supervision_weight": -0.5,
            "metric_supervision_weight": 0.75,
            "prior_supervision_weight": "not-a-number",
        }
    )

    assert weights.semantic == 0.4
    assert weights.expert == 1.0
    assert weights.teacher == 0.0
    assert weights.metric == 0.75
    assert weights.prior == 1.0
