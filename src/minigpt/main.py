"""Main entry point for training and text generation."""

import enum
from typing import Annotated

import torch
from torch import optim
import typer

from minigpt import data
from minigpt import training
from minigpt.models import bigram


class Device(enum.StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    UNDEFINED = "undefined"


def main(
    batch_size: int = typer.Option(32, min=1, help="Number of sequences per batch."),
    block_size: int = typer.Option(8, min=1, help="Sequence length."),
    base_lr: float = typer.Option(6e-4, help="Base learning rate."),
    max_iters: int = typer.Option(3000, min=1, help="Max. training iterations."),
    eval_interval: int = typer.Option(300, min=1, help="Evaluation frequency."),
    eval_iters: int = typer.Option(200, min=1, help="Number of iterations per eval."),
    device: Annotated[
        Device,
        typer.Option(
            help="Device to run training on. Default will attempt to use accelerator"
            " if available, or 'cpu' otherwise."
        ),
    ] = Device.UNDEFINED,
) -> None:
    # NOTE: data and model may be too small to take full advantage of GPU parallelism.
    # CPU-to-GPU overhead is likely significant compared to performance gain on GPU.
    # CPU training could be faster.
    if device == Device.UNDEFINED:
        device_ = torch.accelerator.current_accelerator() or "cpu"
    else:
        device_ = str(device)
    typer.echo(f"Using device: {device_}")

    # prepare data
    text = data.get_text_from_url()
    tokenizer = data.CharacterTokenizer(text)
    dataset = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = data.train_test_split(dataset)
    train_dataloader = data.DataLoader(
        train_data, batch_size=batch_size, block_size=block_size, device=device
    )
    val_dataloader = data.DataLoader(
        val_data, batch_size=batch_size, block_size=block_size, device=device
    )

    # train model
    model = bigram.BigramLanguageModel(tokenizer.vocab_size)
    m = model.to(device_)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    training.train_model(
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


if __name__ == "__main__":
    typer.run(main)
