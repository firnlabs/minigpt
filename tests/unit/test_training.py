"""Tests for training.py."""

import pytest

from minigpt import training


@pytest.mark.parametrize(
    "it, expected_lr",
    [
        (0, 0.0),  # Start of warmup
        (1, 0.01),  # Warmup phase (1 * 1/100)
        (100, 1.0),  # End of warmup (100/100)
        (550, 0.55),  # Midpoint of cosine decay phase
        (1000, 0.1),  # End of decay phase (lr = min_lr effectively)
        (1001, 0.1),  # Post-decay (clamped to min_lr)
    ],
)
def test_get_learning_rate(it, expected_lr):
    base_lr = 1.0
    min_lr = 0.1
    warmup_iters = 100
    lr_decay_iters = 1000

    lr = training._get_learning_rate(
        base_lr=base_lr,
        min_lr=min_lr,
        it=it,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
    )

    assert lr == pytest.approx(expected_lr)
