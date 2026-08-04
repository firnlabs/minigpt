"""Test the CLI interface of the minigpt package."""

import pytest
import typer
from typer.testing import CliRunner

from minigpt.main import main


@pytest.fixture
def cli_app():
    # Dynamically bind main to a Typer app instance for command execution
    app = typer.Typer()
    app.command()(main)
    return app


@pytest.fixture
def mock_text_data() -> str:
    return "the quick brown fox jumps over the lazy dog"


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
