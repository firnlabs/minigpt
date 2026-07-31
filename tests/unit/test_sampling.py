"""Tests for sampling.py."""

import pytest
import torch

from minigpt import sampling


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.0, 0.9, 0.5, 0.1, 0.3])


class TestProcessLogits:
    @pytest.mark.parametrize("temperature", [0.5, 1.0])
    def test_temperature(self, logits, temperature):
        expected = logits / temperature

        actual = sampling._process_logits(logits, temperature)

        assert torch.allclose(actual, expected)


class TestSampleNextToken:
    def test_temperature_zero(self, logits):
        actual = sampling.sample_next_token(logits, temperature=0)

        assert actual == torch.tensor([1])
