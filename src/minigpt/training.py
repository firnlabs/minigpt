"""Model training routines."""

import torch
from torch import optim


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
