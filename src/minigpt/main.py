"""Main entry point for training and text generation."""

import enum
from typing import Annotated
from typing import assert_never

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


def _get_available_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


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
    max_new_tokens: int = typer.Option(
        500, min=1, help="Number of tokens to generate after training."
    ),
    temperature: float = typer.Option(
        1.0, min=0.0, max=1.0, help="Temperature for token generation."
    ),
    top_k: int | None = typer.Option(None, min=1, help="Top-k for token generation."),
    top_p: int | None = typer.Option(None, min=1, help="Top-p for token generation."),
    weight_decay: float = typer.Option(
        0.1,
        min=0.0,
        max=1.0,
        help="For GPT: optimizer weight decay for 2D weights.",
    ),
    beta1: float = typer.Option(
        0.9, min=0.0, max=1.0, help="For GPT: AdamW beta1 (first moment decay)."
    ),
    beta2: float = typer.Option(
        0.95, min=0.0, max=1.0, help="For GPT: AdamW beta2 (second moment decay)."
    ),
) -> None:
    """Trains a character-level language model and generates text."""
    # NOTE: data and model may be too small to take full advantage of GPU parallelism.
    # CPU-to-GPU overhead is likely significant compared to performance gain on GPU.
    # CPU training could be faster.
    device_ = (
        torch.device(device)
        if device != DeviceChoice.UNDEFINED
        else _get_available_device()
    )
    typer.echo(f"Using device: {device_}")

    # prepare data
    text = data.get_text_from_url()
    tokenizer = data.CharacterTokenizer(text)
    dataset = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = data.train_test_split(dataset)
    train_dataloader = data.DataLoader(
        train_data, batch_size=batch_size, block_size=block_size, device=device_
    )
    val_dataloader = data.DataLoader(
        val_data, batch_size=batch_size, block_size=block_size, device=device_
    )

    # get model
    match model_type:
        case ModelChoice.GPT:
            model = gpt.GPT(
                vocab_size=tokenizer.vocab_size,
                num_layers=num_layers,
                num_heads=num_heads,
                block_size=block_size,
                emb_size=emb_size,
                dropout_p=dropout,
            )
        case ModelChoice.BIGRAM:
            model = bigram.BigramLanguageModel(tokenizer.vocab_size)
        case _ as unreachable:  # pragma: no cover
            # This should not happen since typer already validates model_type arg
            assert_never(unreachable)
    typer.echo(f"Using model: {model_type}")

    # train model
    m = model.to(device_)
    if isinstance(m, gpt.GPT):
        optimizer = m.configure_optimizers(
            weight_decay=weight_decay, learning_rate=base_lr, betas=(beta1, beta2)
        )
    else:
        optimizer = optim.AdamW(m.parameters(), lr=base_lr)
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
    context = torch.zeros((1, 1), dtype=torch.long, device=device_)
    generated_tokens = m.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    typer.echo(tokenizer.decode(generated_tokens[0].tolist()))


if __name__ == "__main__":
    typer.run(main)
