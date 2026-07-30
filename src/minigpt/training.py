"""Model training routines."""

import math

import torch
from torch import optim


def _get_learning_rate(
    *, base_lr: float, min_lr: float, it: int, warmup_iters: int, lr_decay_iters: int
) -> float:
    """Calculates learning rate with linear warmup and cosine decay.

    Following the linear warmup (Goyal et al., 2017) and cosine decay
    (Loshchilov & Hutter, 2016) schedule common in LLM training.

    References:
        - Implementation adapted from Andrej Karpathy's nanoGPT:
          https://github.com/karpathy/nanoGPT/blob/master/train.py
        - Goyal et al. (2017): https://arxiv.org/abs/1706.02677
        - Loshchilov & Hutter (2016): https://arxiv.org/abs/1608.03983
    """
    # NOTE: warmup_iters == 0 implies skipping the warmup phase entirely
    if warmup_iters > 0 and it < warmup_iters:
        lr = base_lr * it / warmup_iters
    elif it > lr_decay_iters:
        lr = min_lr
    else:
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (base_lr - min_lr)
    return lr


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
