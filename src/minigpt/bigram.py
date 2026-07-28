"""Bigram language model.

Adapted from Andrej Karpathy's ng-video-lecture:
https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
"""

from urllib import request

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

torch.manual_seed(1337)

_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharacterTokenizer:
    """Character-level tokenizer."""

    def __init__(self, text: str):
        self._chars = sorted(list(set(text)))
        self._vocab_size = len(self._chars)

        # create bidirectional mappings
        self._char_to_idx = {c: i for i, c in enumerate(self._chars)}
        self._idx_to_char = {i: c for c, i in self._char_to_idx.items()}

    def encode(self, text: str) -> list[int]:
        """Converts a text string to a list of integer token IDs."""
        return [self._char_to_idx[c] for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs to a text string."""
        return "".join([self._idx_to_char[idx] for idx in token_ids])


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
        # idx shape: (B, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # focus only on the last timestep
            logits = logits[:, -1, :]  # shape: (B, C)
            probs = F.softmax(logits, dim=-1)  # shape: (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # shape (B, T+1)
        return idx


class DataLoader:
    """Custom data loader."""

    def __init__(
        self,
        dataset: torch.Tensor,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
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


def _get_text_from_url() -> str:
    with request.urlopen(_DATA_URL) as response:
        return response.read().decode("utf-8")


def train_test_split(
    indices: torch.Tensor, train_prop: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_prop * len(indices))
    return indices[:n], indices[n:]


@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, eval_iters: int):
    out = {}
    model.eval()  # set model eval mode (currently no-op)
    for split, dataloader in zip(
        ("train", "val"), (train_dataloader, val_dataloader), strict=True
    ):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(dataloader)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set model train mode (currently no-op)
    return out


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer: optim.Optimizer,
    max_iters,
    eval_interval,
    eval_iters,
) -> None:
    """Trains the model."""
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_dataloader, val_dataloader, eval_iters)
            print(
                f"step {iter}: train loss {losses['train']:.4f},"
                f" val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = next(train_dataloader)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # hyperparameters
    batch_size = 32  # num. independent sequences to process in parallel
    block_size = 8  # max content length for predictions
    vocab_size = 65
    learning_rate = 1e-2
    max_iters = 3000
    eval_interval = 300
    eval_iters = 200

    # NOTE: data and model may be too small to take full advantage of GPU parallelism.
    # CPU-to-GPU overhead is likely significant compared to performance gain on GPU.
    # CPU training could be faster.
    device = torch.accelerator.current_accelerator() or "cpu"
    device = "cpu"
    print(f"Using device: {device}")
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # prepare data
    text = _get_text_from_url()
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_test_split(data)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, block_size=block_size, device=device
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, block_size=block_size, device=device
    )

    # train model
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        max_iters,
        eval_interval,
        eval_iters,
    )

    # generate some text from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
