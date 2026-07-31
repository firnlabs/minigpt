"""Bigram language model.

Adapted from Andrej Karpathy's ng-video-lecture:
https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
"""

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    """Bigram language model."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass."""
        logits = self._token_embedding_table(idx)  # shape: (B, T, C)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # expected idx shape: (B=1, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # focus only on the last timestep
            logits = logits[:, -1, :]  # shape: (B, C)
            probs = F.softmax(logits, dim=-1)  # shape: (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # shape: (B, T+1)
        return idx
