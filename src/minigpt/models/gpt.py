"""Decoder-only generative pretrained transformer (GPT) model."""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# Standard deviation for initialization described in Radford et al. (2018)
# (https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
_INIT_STD = 0.02


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

    def forward(
        self,
        x: torch.Tensor,
        prev_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        assert C == self._emb_size
        assert T <= self._block_size

        c_attn = self._c_attn(x)  # shape: (B, T, 3*C)
        q, k, v = c_attn.split(self._emb_size, dim=2)  # k, q, v shape: (B, T, C)

        # resulting q, k, v shape: (B, NH, T, HS)
        q = q.view(B, T, self._num_heads, self._head_size).transpose(1, 2)
        k = k.view(B, T, self._num_heads, self._head_size).transpose(1, 2)
        v = v.view(B, T, self._num_heads, self._head_size).transpose(1, 2)

        if prev_kv is not None:
            prev_k, prev_v = prev_kv
            k = torch.cat([prev_k, k], dim=2)  # shape: (B, NH, T_total, HS)
            v = torch.cat([prev_v, v], dim=2)  # shape: (B, NH, T_total, HS)

        T_total = k.shape[2]
        assert T_total <= self._block_size
        T_prev = T_total - T

        wei = self._head_size**-0.5 * q @ k.mT  # shape: (B, NH, T, T_total)
        # Apply the mask the last [T_prev, T_total] tokens
        masked_wei = wei.masked_fill(
            self._mask[:, :, T_prev:T_total, :T_total] == 0, float("-inf")
        )
        norm_wei = torch.softmax(masked_wei, dim=-1)
        drop_wei = self._attn_dropout(norm_wei)  # shape: (B, NH, T, T)

        y = drop_wei @ v  # shape: (B, NH, T, HS)

        # re-assemble heads
        y = y.transpose(1, 2)  # shape: (B, T, NH, HS)
        y = y.contiguous().view(B, T, C)  # shape: (B, T, C)

        out = self._resid_dropout(self._c_proj(y))  # shape: (B, T, C)
        kv = (k, v) if return_kv else None
        return out, kv


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


class GPT(nn.Module):
    """Generative pre-trained transfomer model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
        emb_size: int,
        dropout_p: float = 0.5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers
        self._block_size = block_size
        self._wte = nn.Embedding(vocab_size, emb_size)
        self._wpe = nn.Embedding(block_size, emb_size)
        self._drop = nn.Dropout(dropout_p)
        self._h = nn.ModuleList(
            [
                Transformer(
                    num_heads=num_heads,
                    block_size=block_size,
                    emb_size=emb_size,
                    dropout_p=dropout_p,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self._ln_f = nn.LayerNorm(emb_size, bias=bias)
        self._lm_head = nn.Linear(emb_size, vocab_size, bias=False)

        # share weights between input token embeddings and output lm_head projection
        self._lm_head.weight = self._wte.weight

        # Flag residual layers for specific initialization
        for name, module in self.named_modules():
            if name.endswith("_c_proj"):
                module._is_residual = True

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # use default standard deviation
            std = _INIT_STD

            # scale standard deviation for residual projection layers
            # based on Radford et al. (2019)
            # https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
            if getattr(module, "_is_residual", False):
                std *= (2 * self._num_layers) ** -0.5

            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)

        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            pass  # no-op

    def configure_optimizers(
        self, *, weight_decay: float, learning_rate: float, betas: tuple[float, float]
    ) -> optim.Optimizer:
        # Fetch all parameters that require gradients
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}

        # --- split parameters based on dimension ---
        # 2D weights of Linear and Embedding layers
        decay_params = [p for _, p in sorted(params.items()) if p.dim() >= 2]
        # 1D biases, LayerNorm
        no_decay_params = [p for _, p in sorted(params.items()) if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self._block_size

        # forward token and positional embeddings
        token_emb = self._wte(idx)
        device = idx.device
        pos = torch.arange(T, dtype=torch.long, device=device)
        pos_emb = self._wpe(pos)
        x = self._drop(token_emb + pos_emb)

        # forward through transformer blocks
        for block in self._h:
            x = block(x)

        # forward final layer norm
        x = self._ln_f(x)

        if targets is not None:  # training
            logits = self._lm_head(x)
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
        else:  # inference
            # predict on last token in sequence only
            logits = self._lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Crop `idx`` to the last `block_size` tokens (i.e. context window)
            idx_gen = idx[:, -self._block_size :]
            logits, _ = self(idx_gen)
            # focus on the logit for the last generated token
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
