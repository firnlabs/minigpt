"""Decoder-only generative pretrained transformer (GPT) model."""

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Causal self-attention block."""

    def __init__(
        self, *, num_heads, block_size, emb_size, dropout_p=0.5, bias=False
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._emb_size = emb_size
        self._head_size = emb_size // num_heads
        self._block_size = block_size

        # combine query, key and value into one layer
        self._c_attn = nn.Linear(emb_size, 3 * emb_size, bias=bias)

        # add linear output layer
        self._c_proj = nn.Linear(emb_size, emb_size, bias=bias)

        # dropout layers
        self._attn_dropout = nn.Dropout(dropout_p)
        self._resid_dropout = nn.Dropout(dropout_p)

        # causal mask
        self._mask = nn.Buffer(
            torch.tril(torch.ones(block_size, block_size)).view(
                (1, 1, block_size, block_size)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self._emb_size
        assert T <= self._block_size

        c_attn = self._c_attn(x)  # shape: (B, T, 3*C)
        q, k, v = c_attn.split(self._emb_size, dim=2)  # k, q, v shape: (B, T, C)

        # resulting q, k, v shape: (B, NH, T, HS)
        q = q.view(B, T, self._num_heads, self._head_size).transpose(1, 2)
        k = k.view(B, T, self._num_heads, self._head_size).transpose(1, 2)
        v = v.view(B, T, self._num_heads, self._head_size).transpose(1, 2)

        wei = self._head_size**-0.5 * q @ k.mT  # shape: (B, NH, T, T)
        masked_wei = wei.masked_fill(self._mask[:, :, :T, :T] == 0, float("-inf"))
        norm_wei = torch.softmax(masked_wei, dim=-1)
        drop_wei = self._attn_dropout(norm_wei)  # shape: (B, NH, T, T)

        y = drop_wei @ v  # shape: (B, NH, T, HS)

        # re-assemble heads
        y = y.transpose(1, 2)  # shape: (B, T, NH, HS)
        y = y.contiguous().view(B, T, C)  # shape: (B, T, C)

        return self._resid_dropout(self._c_proj(y))  # shape: (B, T, C)


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network) block."""

    def __init__(
        self, emb_size: int, dropout_p: float = 0.5, bias: bool = False
    ) -> None:
        super().__init__()
        self._c_fc = nn.Linear(emb_size, 4 * emb_size, bias=bias)  # C -> 4C
        self._gelu = nn.GELU()  # 4C -> 4C
        self._c_proj = nn.Linear(4 * emb_size, emb_size, bias=bias)  # 4C -> C
        self._dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._c_fc(x)
        x = self._gelu(x)
        x = self._c_proj(x)
        x = self._dropout(x)
        return x


class Transformer(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        *,
        num_heads: int,
        block_size: int,
        emb_size: int,
        dropout_p: float = 0.5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._ln1 = nn.LayerNorm(emb_size, bias=bias)
        self._attn = CausalSelfAttention(
            num_heads=num_heads,
            block_size=block_size,
            emb_size=emb_size,
            dropout_p=dropout_p,
            bias=bias,
        )
        self._ln2 = nn.LayerNorm(emb_size, bias=bias)
        self._mlp = MLP(emb_size, dropout_p, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use pre-LN residual connections
        x = x + self._attn(self._ln1(x))
        x = x + self._mlp(self._ln2(x))
        return x
