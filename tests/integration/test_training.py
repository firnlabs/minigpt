"""Integration test for training.py."""

from dataclasses import dataclass

import pytest
from pytest_mock import MockerFixture
import torch
from torch import optim

from minigpt import data
from minigpt import training
from minigpt.models import bigram
from minigpt.models import gpt


@dataclass
class HParams:
    vocab_size: int = 8
    dataset_size: int = 32
    batch_size: int = 2
    block_size: int = 4
    learning_rate: float = 1e-2
    max_iters: int = 4
    eval_interval: int = 2
    eval_iters: int = 2
    train_prop: float = 0.75

    # GPT-specific
    num_layers: int = 2
    num_heads: int = 2
    emb_size: int = 4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)


@pytest.fixture
def hparams() -> HParams:
    return HParams()


@pytest.fixture
def dataset(hparams: HParams) -> torch.Tensor:
    return torch.randint(high=hparams.vocab_size, size=(hparams.dataset_size,))


@pytest.fixture
def train_val_dataloaders(
    dataset: torch.Tensor, hparams: HParams
) -> tuple[data.DataLoader, data.DataLoader]:
    train_data, val_data = data.train_test_split(dataset, train_prop=hparams.train_prop)
    train_dataloader = data.DataLoader(
        train_data, hparams.batch_size, hparams.block_size
    )
    val_dataloader = data.DataLoader(val_data, hparams.batch_size, hparams.block_size)
    return train_dataloader, val_dataloader


@pytest.fixture
def bigram_model(hparams: HParams) -> bigram.BigramLanguageModel:
    return bigram.BigramLanguageModel(hparams.vocab_size)


@pytest.fixture
def gpt_model(hparams: HParams) -> gpt.GPT:
    return gpt.GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        block_size=hparams.block_size,
        emb_size=hparams.emb_size,
    )


@pytest.mark.parametrize("model_fixture_name", ["bigram_model", "gpt_model"])
def test_train_model(
    hparams: HParams,
    train_val_dataloaders: tuple[data.DataLoader, data.DataLoader],
    model_fixture_name: str,
    request: pytest.FixtureRequest,
    mocker: MockerFixture,
):
    train_dataloader, val_dataloader = train_val_dataloaders
    model = request.getfixturevalue(model_fixture_name)
    is_gpt = model_fixture_name == "gpt_model"
    if is_gpt:
        mock_get_learning_rate = mocker.patch(
            "minigpt.training._get_learning_rate", return_value=hparams.learning_rate
        )
        optimizer = model.configure_optimizers(
            weight_decay=hparams.weight_decay,
            learning_rate=hparams.learning_rate,
            betas=hparams.betas,
        )
    else:  # implies bigram model
        optimizer = optim.AdamW(model.parameters(), lr=hparams.learning_rate)
    params_before = [p.clone() for p in model.parameters()]

    training.train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        hparams.max_iters,
        hparams.eval_interval,
        hparams.eval_iters,
        use_lr_scheduler=is_gpt,
    )

    if is_gpt:
        mock_get_learning_rate.assert_called()

    params_changed = any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, model.parameters(), strict=True)
    )
    assert params_changed
