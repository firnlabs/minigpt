"""Tests for models/gpt.py."""

import torch

from minigpt.models import gpt


class TestCausalSelfAttention:
    def test_forward(self):
        num_heads = 2
        block_size = 8
        emb_size = 32
        batch_size = 4
        attn = gpt.CausalSelfAttention(
            num_heads=num_heads, block_size=block_size, emb_size=emb_size
        )
        x = torch.rand((batch_size, block_size, emb_size))

        out = attn.forward(x)

        assert not torch.allclose(out, x)
        assert out.shape == (batch_size, block_size, emb_size)


class TestMLP:
    def test_forward(self):
        block_size = 8
        emb_size = 32
        batch_size = 4
        mlp = gpt.MLP(emb_size=32)
        x = torch.rand((batch_size, block_size, emb_size))

        out = mlp.forward(x)

        assert out.shape == (batch_size, block_size, emb_size)
        assert not torch.allclose(out, x)


class TestTransfomer:
    def test_forward(self):
        num_heads = 2
        block_size = 8
        emb_size = 32
        batch_size = 4
        transformer = gpt.Transformer(
            num_heads=num_heads, block_size=block_size, emb_size=emb_size
        )
        x = torch.rand((batch_size, block_size, emb_size))

        out = transformer.forward(x)

        assert out.shape == (batch_size, block_size, emb_size)
        assert not torch.allclose(out, x)
