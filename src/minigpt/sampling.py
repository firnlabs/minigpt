"""Token sampling."""

import torch
from torch.nn import functional as F


def _process_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
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

    # select logits by top_p if specified
    if top_p is not None:
        assert top_p > 0.0
        assert top_p <= 1.0

        # sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # get sorted probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # get cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # find indices to remove where cumulative probability exceeds top_p
        remove_mask = cumulative_probs > top_p

        # shift right to keep indices whose cumululative probability is just above top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()

        # mark the first index to remove as False to keep it
        # (never remove the highest probability token)
        remove_mask[..., 0] = False

        # map remove masks to the original indices of the logits
        indices_to_remove = remove_mask.scatter(
            dim=-1, index=sorted_indices, src=remove_mask
        )

        # set logits to -inf for tokens to remove
        logits[indices_to_remove] = float("-inf")

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature == 0:
        # use greedy sampling
        return torch.argmax(logits, dim=-1, keepdim=True)

    processed_logits = _process_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    probs = F.softmax(processed_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
