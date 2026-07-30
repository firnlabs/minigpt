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


class TestGPT:
    def test_forward_no_targets(self):
        vocab_size = 4
        num_layers = 2
        num_heads = 2
        block_size = 8
        emb_size = 32
        batch_size = 4
        vocab_size = 4
        gpt_model = gpt.GPT(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            block_size=block_size,
            emb_size=emb_size,
        )
        idx = torch.randint(high=vocab_size, size=(batch_size, block_size))

        logits, loss = gpt_model.forward(idx)

        assert logits.shape == (batch_size, 1, vocab_size)  # predict on last token only
        assert loss is None

    def test_forward_with_targets(self):
        vocab_size = 4
        num_layers = 2
        num_heads = 2
        block_size = 8
        emb_size = 32
        batch_size = 4
        vocab_size = 4
        gpt_model = gpt.GPT(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            block_size=block_size,
            emb_size=emb_size,
        )
        idx = torch.randint(high=vocab_size, size=(batch_size, block_size))
        targets = torch.randint(high=vocab_size, size=(batch_size, block_size))

        logits, loss = gpt_model.forward(idx, targets)

        assert logits.shape == (batch_size, block_size, vocab_size)
        assert loss is not None
        assert loss.shape == ()
