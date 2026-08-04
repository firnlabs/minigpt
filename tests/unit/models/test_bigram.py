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

    def test_generate(self, model, vocab_size, mocker):
        max_new_tokens = 2
        x = torch.randint(high=vocab_size, size=(1, 1))
        temperature = 0.5
        top_k = 2
        top_p = 0.8
        mock_sample_next_token = mocker.patch(
            "minigpt.sampling.sample_next_token",
            return_value=torch.randint(high=vocab_size, size=(1, 1)),
        )

        out = model.generate(
            x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p
        )

        assert out.shape == (1, 3)
        assert mock_sample_next_token.call_count == max_new_tokens
        first_call = mock_sample_next_token.call_args_list[0]
        assert first_call == mocker.call(
            mocker.ANY, temperature=temperature, top_k=top_k, top_p=top_p
        )
