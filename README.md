# miniGPT: Minimal decoder-only transformer

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-99.5%25-brightgreen.svg)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A minimal, modular implementation of a decoder-only generative pre-trained transformer (GPT) and bigram baseline written in PyTorch. Inspired by [karpathy/nanogpt](https://github.com/karpathy/nanogpt).

---

## Getting started

Prerequisites: Python 3.11+ and [uv](https://github.com/astral-sh/uv).

Install dependencies:

```bash
uv sync
```

### Training and text generation

Train the GPT model on the Tiny Shakespeare dataset and generate completions:

```bash
uv run python src/minigpt/main.py --model-type gpt --max-iters 1000 --batch-size 32
```

Or run the lightweight bigram baseline model:

```bash
uv run python src/minigpt/main.py --model-type bigram --max-iters 500
```

---

## Directory structure

```text
.
├── src/minigpt/
│   ├── models/
│   │   ├── bigram.py           # Baseline bigram lookup language model
│   │   └── gpt.py              # Causal self-attention, transformer blocks, & GPT
│   ├── data.py                 # Character tokenizer & dataloaders
│   ├── main.py                 # Typer CLI for training and text generation
│   ├── sampling.py             # Autoregressive generation (temperature, top-k/top-p)
│   └── training.py             # Training loop, evaluation, & learning rate schedule
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── uv.lock
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

---

## Contributing

Contributions, feedback, and bug reports are welcome. To contribute:

1. Fork the repository and create a feature branch (e.g. `git checkout -b feat/my-feature`).
2. Verify all tests pass and linting succeeds (`uv run pytest && uv run ruff check`).
3. Submit a pull request describing the rationale, changes made, and invariants preserved.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
