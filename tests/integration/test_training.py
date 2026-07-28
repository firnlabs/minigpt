"""Integration test for training.py."""

import torch
from torch import optim

from minigpt import data
from minigpt import training
from minigpt.models import bigram


def test_train_model():
    # hyperparams
    vocab_size = 8
    dataset_size = 32
    batch_size = 2
    block_size = 4
    learning_rate = 1e-2
    max_iters = 4
    eval_interval = 2
    eval_iters = 2

    # setup
    dataset = torch.randint(high=vocab_size, size=(dataset_size,))
    train_data, val_data = data.train_test_split(dataset, train_prop=0.75)
    train_dataloader = data.DataLoader(train_data, batch_size, block_size)
    val_dataloader = data.DataLoader(val_data, batch_size, block_size)
    model = bigram.BigramLanguageModel(vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    weights_before = model._token_embedding_table.weight.clone()

    training.train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        max_iters,
        eval_interval,
        eval_iters,
    )

    weights_after = model._token_embedding_table.weight

    assert not torch.allclose(weights_before, weights_after)
