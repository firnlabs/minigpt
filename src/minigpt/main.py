"""Main entry point for training and text generation."""

import torch
from torch import optim

from minigpt import data
from minigpt import training
from minigpt.models import bigram

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
    model = bigram.BigramLanguageModel(vocab_size)
    m = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

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
