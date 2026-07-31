"""Token sampling."""

import torch
from torch.nn import functional as F


def _process_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    assert temperature > 0
    assert temperature <= 1

    if temperature != 1.0:
        logits = logits / temperature

    return logits


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        # use greedy sampling
        return torch.argmax(logits, dim=-1, keepdim=True)

    processed_logits = _process_logits(logits, temperature=temperature)
    probs = F.softmax(processed_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
