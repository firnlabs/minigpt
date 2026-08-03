"""Token sampling."""

import torch
from torch.nn import functional as F


def _process_logits(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    assert temperature > 0
    assert temperature <= 1

    # scale logits by temperature
    if temperature != 1.0:
        logits = logits / temperature

    # select logits by top_k if specified
    if top_k is not None:
        assert top_k > 0
        assert top_k <= logits.size(-1)
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[..., -1, None]] = float("-inf")

    return logits


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    if temperature == 0:
        # use greedy sampling
        return torch.argmax(logits, dim=-1, keepdim=True)

    processed_logits = _process_logits(logits, temperature=temperature, top_k=top_k)
    probs = F.softmax(processed_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
