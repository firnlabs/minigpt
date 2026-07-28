"""Tests for models/bigram.py."""

import pytest
import torch

from minigpt.models import bigram


@pytest.fixture
def vocab_size() -> int:
    return 100


@pytest.fixture
def model(vocab_size) -> bigram.BigramLanguageModel:
    return bigram.BigramLanguageModel(vocab_size=vocab_size)


class TestBigramLanguageModel:
    def test_forward_no_targets(self, model, vocab_size):
        batch_size = 4
        block_size = 8
        x = torch.randint(high=vocab_size, size=(batch_size, block_size))

        logits, loss = model.forward(x)

        assert logits.shape == (batch_size, block_size, vocab_size)
        assert loss is None

    def test_forward_with_targets(self, model, vocab_size):
        batch_size = 4
        block_size = 8
        x = torch.randint(high=vocab_size, size=(batch_size, block_size))
        y = torch.randint(high=vocab_size, size=(batch_size, block_size))

        logits, loss = model.forward(x, targets=y)

        assert logits.shape == (batch_size * block_size, vocab_size)
        assert loss.shape == ()

    def test_generate(self, model, vocab_size):
        max_new_tokens = 2
        x = torch.randint(high=vocab_size, size=(1, 1))

        out = model.generate(x, max_new_tokens)

        assert out.shape == (1, 3)
