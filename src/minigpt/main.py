"""Main entry point for training and text generation."""

import enum
from typing import Annotated

import torch
from torch import optim
import typer

from minigpt import data
from minigpt import training
from minigpt.models import bigram
from minigpt.models import gpt


class DeviceChoice(enum.StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    UNDEFINED = "undefined"


class ModelChoice(enum.StrEnum):
    BIGRAM = "bigram"
    GPT = "gpt"


def main(
    model_type: Annotated[
        ModelChoice, typer.Option(help="Model to use.")
    ] = ModelChoice.GPT,
    batch_size: int = typer.Option(32, min=1, help="Number of sequences per batch."),
    block_size: int = typer.Option(64, min=1, help="Sequence length."),
    base_lr: float = typer.Option(1e-3, help="Base learning rate."),
    max_iters: int = typer.Option(1000, min=1, help="Max. training iterations."),
    eval_interval: int = typer.Option(300, min=1, help="Evaluation frequency."),
    eval_iters: int = typer.Option(200, min=1, help="Number of iterations per eval."),
    device: Annotated[
        DeviceChoice,
        typer.Option(
            help="Device to run training on. Default will attempt to use accelerator"
            " if available, or 'cpu' otherwise."
        ),
    ] = DeviceChoice.UNDEFINED,
    num_layers: int = typer.Option(
        4, min=1, help="For GPT: number of transfomer blocks."
    ),
    num_heads: int = typer.Option(
        4, min=1, help="For GPT: number of attention heads per transformer block."
    ),
    emb_size: int = typer.Option(64, min=1, help="For GPT: embedding dimension."),
    dropout: float = typer.Option(0.1, min=0.0, max=1.0, help="For GPT: dropout %."),
) -> None:
    # NOTE: data and model may be too small to take full advantage of GPU parallelism.
    # CPU-to-GPU overhead is likely significant compared to performance gain on GPU.
    # CPU training could be faster.
    if device == DeviceChoice.UNDEFINED:
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

    # get model
    if model_type == ModelChoice.GPT:
        typer.echo("Using GPT model.")
        model = gpt.GPT(
            vocab_size=tokenizer.vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            block_size=block_size,
            emb_size=emb_size,
            dropout_p=dropout,
        )
    elif model_type == ModelChoice.BIGRAM:
        typer.echo("Using bigram model.")
        model = bigram.BigramLanguageModel(tokenizer.vocab_size)

    # train model
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
        use_lr_scheduler=model_type == ModelChoice.GPT,
    )

    # generate some text from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    typer.run(main)
