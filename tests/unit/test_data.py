import pytest
import torch

from minigpt import data


def test_get_text_from_url(mocker):
    mock_urlopen = mocker.patch("urllib.request.urlopen")
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = b"some text"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    text = data.get_text_from_url()

    assert mock_urlopen.called_once()
    assert text == "some text"


@pytest.fixture
def tokenizer() -> data.CharacterTokenizer:
    text = "abcde"
    return data.CharacterTokenizer(text)


class TestCharacterTokenizer:
    def test_encode(self, tokenizer):
        assert tokenizer.encode("dcba") == [3, 2, 1, 0]

    def test_decode(self, tokenizer):
        assert tokenizer.decode([1, 4, 0, 2]) == "beac"


@pytest.fixture
def indices(size=100) -> torch.Tensor:
    return torch.arange(size)


@pytest.mark.parametrize("train_prop", [1.0, 0.9])
def test_train_test_split(indices, train_prop):
    train, val = data.train_test_split(indices, train_prop)

    assert len(train) == 100 * train_prop
    assert len(val) == pytest.approx(100 * (1.0 - train_prop))


@pytest.fixture
def dataloader_factory():
    dataset = torch.randint(high=4, size=(64,))

    def _create_dataloader(batch_size: int, block_size: int) -> data.DataLoader:
        return data.DataLoader(dataset, batch_size=batch_size, block_size=block_size)

    return _create_dataloader


class TestDataLoader:
    @pytest.mark.parametrize("batch_size, block_size", [(4, 8), (16, 4)])
    def test_get_next_batch(self, dataloader_factory, batch_size, block_size):
        dataloader = dataloader_factory(batch_size=batch_size, block_size=block_size)

        xb, yb = next(dataloader)

        assert xb.shape == (batch_size, block_size)
        assert yb.shape == (batch_size, block_size)
