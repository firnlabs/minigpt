"""Tests for models/gpt.py."""

from dataclasses import asdict
from dataclasses import dataclass

import pytest
import torch

from minigpt.models import gpt


@dataclass(frozen=True)
class Config:
    vocab_size: int = 4
    num_layers: int = 2
    num_heads: int = 2
    block_size: int = 8
    emb_size: int = 32
    batch_size: int = 4


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def hidden(config) -> torch.Tensor:
    return torch.rand((config.batch_size, config.block_size, config.emb_size))


@pytest.fixture
def idx(config) -> torch.Tensor:
    return torch.randint(
        high=config.vocab_size,
        size=(config.batch_size, config.block_size),
    )


@pytest.fixture
def targets(config) -> torch.Tensor:
    return torch.randint(
        high=config.vocab_size,
        size=(config.batch_size, config.block_size),
    )


@pytest.fixture
def gpt_model(config) -> gpt.GPT:
    config_dict = asdict(config)
    config_dict.pop("batch_size")
    return gpt.GPT(**config_dict)


class TestCausalSelfAttention:
    def test_forward(self, config, hidden):
        attn = gpt.CausalSelfAttention(
            num_heads=config.num_heads,
            block_size=config.block_size,
            emb_size=config.emb_size,
        )

        out = attn.forward(hidden)

        assert not torch.allclose(out, hidden)
        assert out.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )


class TestMLP:
    def test_forward(self, config, hidden):
        mlp = gpt.MLP(emb_size=config.emb_size)

        out = mlp.forward(hidden)

        assert out.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )
        assert not torch.allclose(out, hidden)


class TestTransfomer:
    def test_forward(self, config, hidden):
        transformer = gpt.Transformer(
            num_heads=config.num_heads,
            block_size=config.block_size,
            emb_size=config.emb_size,
        )

        out = transformer.forward(hidden)

        assert out.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )
        assert not torch.allclose(out, hidden)


class TestGPT:
    def test_init_weights(self):
        pass  # TODO: implement

    def test_forward_no_targets(self, gpt_model, config, idx):
        logits, loss = gpt_model.forward(idx)

        # predict on last token only
        assert logits.shape == (
            config.batch_size,
            1,
            config.vocab_size,
        )
        assert loss is None

    def test_forward_with_targets(self, gpt_model, config, idx, targets):
        logits, loss = gpt_model.forward(idx, targets)

        assert logits.shape == (
            config.batch_size,
            config.block_size,
            config.vocab_size,
        )
        assert loss is not None
        assert loss.shape == ()
