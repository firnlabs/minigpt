"""Data preparation for model training."""

from urllib import request

import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_text_from_url(url=TINY_SHAKESPEARE_URL) -> str:
    with request.urlopen(url) as response:
        return response.read().decode("utf-8")


class CharacterTokenizer:
    """Character-level tokenizer."""

    def __init__(self, text: str):
        self._chars = sorted(list(set(text)))
        self._vocab_size = len(self._chars)

        # create bidirectional mappings
        self._char_to_idx = {c: i for i, c in enumerate(self._chars)}
        self._idx_to_char = {i: c for c, i in self._char_to_idx.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        """Converts a text string into token IDs."""
        return [self._char_to_idx[c] for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs to a text string."""
        return "".join([self._idx_to_char[idx] for idx in token_ids])


def train_test_split(
    indices: torch.Tensor, train_prop: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_prop * len(indices))
    return indices[:n], indices[n:]


class DataLoader:
    """Custom data loader."""

    def __init__(
        self,
        dataset: torch.Tensor,
        batch_size: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize."""
        self._dataset = dataset.to(device)
        self._batch_size = batch_size
        self._block_size = block_size
        self._device = device

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._dataset
        ix = torch.randint(
            len(data) - self._block_size, (self._batch_size,), device=self._device
        )

        # create grid of offsets of shape (block_size, )
        # i.e. [0, 1, 2, ..., block_size - 1]
        offsets = torch.arange(self._block_size, device=self._device)

        # broadcast offsets to shape (batch_size, block_size)
        idx_x = ix.unsqueeze(1) + offsets
        idx_y = idx_x + 1

        # gather data needed for the batch
        x = data[idx_x]
        y = data[idx_y]
        return x, y
