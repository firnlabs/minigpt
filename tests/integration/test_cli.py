"""Integration tests for CLI execution."""

import pytest
import torch
import typer
from typer.testing import CliRunner

from minigpt import main


@pytest.fixture
def cli_app():
    # Dynamically bind main to a Typer app instance for command execution
    app = typer.Typer()
    app.command()(main.main)
    return app


@pytest.fixture
def mock_text_data() -> str:
    return "the quick brown fox jumps over the lazy dog"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_get_available_device(device, mocker):
    mock_cuda_is_available = mocker.patch("torch.cuda.is_available")
    mock_mps_is_available = mocker.patch("torch.backends.mps.is_available")
    match device:
        case "cuda":
            mock_cuda_is_available.return_value = True
            mock_mps_is_available.return_value = False
        case "mps":
            mock_cuda_is_available.return_value = False
            mock_mps_is_available.return_value = True
        case "cpu":
            mock_cuda_is_available.return_value = False
            mock_mps_is_available.return_value = False

    actual = main._get_available_device()

    assert actual == torch.device(device)


@pytest.mark.parametrize("model_type", ["gpt", "bigram"])
@pytest.mark.parametrize("device", ["undefined", "cpu"])
def test_main_cli_execution(model_type, device, mock_text_data, cli_app, mocker):
    mock_get_text = mocker.patch(
        "minigpt.main.data.get_text_from_url", return_value=mock_text_data
    )
    runner = CliRunner()

    # Run the CLI command with small parameters
    result = runner.invoke(
        cli_app,
        [
            "--model-type",
            model_type,
            "--batch-size",
            "2",
            "--block-size",
            "4",
            "--max-iters",
            "2",
            "--eval-interval",
            "1",
            "--eval-iters",
            "1",
            "--device",
            device,
            "--num-layers",
            "1",
            "--num-heads",
            "1",
            "--emb-size",
            "4",
            "--max-new-tokens",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "Using device:" in result.stdout
    assert f"Using model: {model_type}" in result.stdout
    mock_get_text.assert_called_once()
