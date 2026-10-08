import pytest
import torch

from minigpt import sampling

_NEG_INF = float("-inf")


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.05, 0.9, 0.5, 0.1, 0.3])


class TestProcessLogits:
    @pytest.mark.parametrize("temperature", [0.5, 1.0])
    def test_temperature(self, logits, temperature):
        expected = logits / temperature

        actual = sampling._process_logits(logits, temperature)

        assert torch.allclose(actual, expected)

    def test_top_k_none(self, logits):
        actual = sampling._process_logits(logits, top_k=None)

        assert torch.allclose(actual, logits)

    def test_top_k(self, logits):
        top_k = 3

        actual = sampling._process_logits(logits, top_k=top_k)

        assert torch.allclose(actual, torch.tensor([_NEG_INF, 0.9, 0.5, _NEG_INF, 0.3]))

    def test_top_p_none(self, logits):
        actual = sampling._process_logits(logits, top_p=None)

        assert torch.allclose(actual, logits)

    def test_top_p(self, logits):
        top_p = 0.5

        actual = sampling._process_logits(logits, top_p=top_p)

        assert torch.allclose(
            actual, torch.tensor([_NEG_INF, 0.9, 0.5, _NEG_INF, _NEG_INF])
        )


class TestSampleNextToken:
    def test_temperature_zero(self, logits):
        actual = sampling.sample_next_token(logits, temperature=0)

        assert actual == torch.tensor([1])

    def test_temperature_nonzero(self, logits):
        temperature = 0.5

        actual = sampling.sample_next_token(logits, temperature=temperature)

        assert actual.shape == (1,)
        assert actual.item() in range(len(logits))

    def test_top_k(self, logits):
        top_k = 4
        temperature = 1.0

        actuals = set()
        for _ in range(10):
            actual = sampling.sample_next_token(
                logits, temperature=temperature, top_k=top_k
            )
            actuals.add(actual.item())

        excluded = logits.argmin().item()
        assert excluded not in actuals

    def test_top_p(self, logits):
        top_p = 0.5
        temperature = 1.0

        actuals = set()
        for _ in range(10):
            actual = sampling.sample_next_token(
                logits, temperature=temperature, top_p=top_p
            )
            actuals.add(actual.item())

        excluded_tokens = {0, 3, 4}
        assert excluded_tokens.isdisjoint(actuals)
