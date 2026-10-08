from dataclasses import asdict
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch import optim

from minigpt.models import gpt


@dataclass(frozen=True)
class Config:
    vocab_size: int = 4
    num_layers: int = 2
    num_heads: int = 2
    block_size: int = 8
    emb_size: int = 32
    batch_size: int = 4


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def hidden(config) -> torch.Tensor:
    return torch.rand((config.batch_size, config.block_size, config.emb_size))


@pytest.fixture
def idx(config) -> torch.Tensor:
    return torch.randint(
        high=config.vocab_size,
        size=(config.batch_size, config.block_size),
    )


@pytest.fixture
def targets(config) -> torch.Tensor:
    return torch.randint(
        high=config.vocab_size,
        size=(config.batch_size, config.block_size),
    )


@pytest.fixture
def gpt_model(config) -> gpt.GPT:
    config_dict = asdict(config)
    config_dict.pop("batch_size")
    torch.manual_seed(42)
    model = gpt.GPT(**config_dict)
    model.eval()
    return model


@pytest.fixture
def T_prev(config) -> int:
    return config.block_size // 2


@pytest.fixture
def head_size(config) -> int:
    return config.emb_size // config.num_heads


@pytest.fixture
def prev_k(config, T_prev, head_size) -> torch.Tensor:
    return torch.rand((config.batch_size, config.num_heads, T_prev, head_size))


@pytest.fixture
def prev_v(prev_k) -> torch.Tensor:
    return torch.randn_like(prev_k)


@pytest.fixture
def attention(config) -> gpt.CausalSelfAttention:
    return gpt.CausalSelfAttention(
        num_heads=config.num_heads,
        block_size=config.block_size,
        emb_size=config.emb_size,
    )


@pytest.fixture
def transformer(config) -> gpt.Transformer:
    return gpt.Transformer(
        num_heads=config.num_heads,
        block_size=config.block_size,
        emb_size=config.emb_size,
    )


class TestCausalSelfAttention:
    def test_forward(self, attention, config, hidden):
        logits, kv = attention.forward(hidden)

        assert not torch.allclose(logits, hidden)
        assert logits.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )
        assert kv is None

    def test_forward_with_kv_cache(
        self,
        attention,
        config,
        hidden,
        prev_k,
        prev_v,
        T_prev,
        head_size,
    ):
        hidden_gen = hidden[:, :1, :]  # assume decoding phase

        out, kv = attention.forward(
            hidden_gen, prev_kv=(prev_k, prev_v), use_cache=True
        )

        assert out.shape == (config.batch_size, 1, config.emb_size)
        assert isinstance(kv, tuple)
        k, v = kv
        exp_kv_shape = (config.batch_size, config.num_heads, T_prev + 1, head_size)
        assert k.shape == exp_kv_shape
        assert v.shape == exp_kv_shape


class TestMLP:
    def test_forward(self, config, hidden):
        mlp = gpt.MLP(emb_size=config.emb_size)

        out = mlp.forward(hidden)

        assert out.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )
        assert not torch.allclose(out, hidden)


class TestTransfomer:
    def test_forward(self, transformer, config, hidden):
        out, kv = transformer.forward(hidden)

        assert out.shape == (
            config.batch_size,
            config.block_size,
            config.emb_size,
        )
        assert not torch.allclose(out, hidden)
        assert kv is None

    def test_forward_with_kv_cache(self, transformer, config, hidden, prev_k, prev_v):
        hidden_gen = hidden[:, :1, :]
        out, kv = transformer.forward(
            hidden_gen, prev_kv=(prev_k, prev_v), use_cache=True
        )

        assert out.shape == (config.batch_size, 1, config.emb_size)
        assert not torch.allclose(out, hidden_gen)
        assert isinstance(kv, tuple)


class TestGPT:
    @pytest.mark.parametrize("bias, is_residual", [(False, False), (True, True)])
    def test_init_weights_linear(self, gpt_model, config, bias, is_residual):
        module = nn.Linear(config.emb_size, config.emb_size, bias=bias)
        module._is_residual = is_residual

        gpt_model._init_weights(module)

        assert torch.allclose(module.weight.mean(), torch.tensor(0.0), atol=1e-2)
        if is_residual:
            exp_std = 0.02 * (2 * config.num_layers) ** -0.5
        else:
            exp_std = 0.02
        assert torch.allclose(module.weight.std(), torch.tensor(exp_std), atol=1e-2)
        if bias:
            assert module.bias.all() == 0.0
        else:
            assert module.bias is None

    def test_init_weights_embedding(self, gpt_model, config):
        module = nn.Embedding(config.vocab_size, config.emb_size)

        gpt_model._init_weights(module)

        assert torch.allclose(module.weight.mean(), torch.tensor(0.0), atol=1e-2)
        assert torch.allclose(module.weight.std(), torch.tensor(0.02), atol=1e-2)

    def test_init_weights_layernorm(self, gpt_model, config):
        module = nn.LayerNorm(config.emb_size, bias=True)

        gpt_model._init_weights(module)

        assert module.bias.all() == 0.0

    def test_configure_optimizers(self, gpt_model):
        # Values derived from Brown et al. (2020)
        # (https://arxiv.org/pdf/2005.14165)
        learning_rate = 6e-4
        weight_decay = 0.1
        betas = (0.9, 0.95)

        optimizer: optim.Optimizer = gpt_model.configure_optimizers(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )

        state = optimizer.state_dict()
        groups = state["param_groups"]
        assert len(groups) == 2
        assert groups[0]["lr"] == learning_rate
        assert groups[0]["weight_decay"] == weight_decay
        assert groups[1]["lr"] == learning_rate
        assert groups[1]["weight_decay"] == 0.0

    def test_forward_no_targets(self, gpt_model, config, idx):
        logits, loss, kvs = gpt_model.forward(idx)

        # predict on last token only
        assert logits.shape == (
            config.batch_size,
            1,
            config.vocab_size,
        )
        assert loss is None
        assert kvs is None

    def test_forward_with_targets(self, gpt_model, config, idx, targets):
        logits, loss, kvs = gpt_model.forward(idx, targets)

        assert logits.shape == (
            config.batch_size,
            config.block_size,
            config.vocab_size,
        )
        assert loss is not None
        assert loss.shape == ()
        assert kvs is None

    def test_forward_with_kv_cache(
        self, gpt_model, config, idx, prev_k, prev_v, T_prev, head_size
    ):
        prev_kvs = [(prev_k, prev_v) for _ in range(config.num_layers)]
        idx_gen = idx[:, :1]

        logits, loss, kvs = gpt_model.forward(
            idx_gen, prev_kvs=prev_kvs, use_cache=True
        )

        assert logits.shape == (
            config.batch_size,
            1,
            config.vocab_size,
        )
        assert loss is None
        assert kvs is not None
        assert len(kvs) == config.num_layers
        for kv in kvs:
            for t in kv:
                assert t.shape == (
                    config.batch_size,
                    config.num_heads,
                    T_prev + 1,
                    head_size,
                )

    @pytest.mark.parametrize("use_cache", [True, False])
    def test_generate(self, gpt_model, config, use_cache, mocker):
        input_length = config.block_size + 1  # input larger than context window
        max_new_tokens = 3
        idx = torch.randint(high=config.vocab_size, size=(1, input_length))
        temperature = 0.5
        top_k = 2
        top_p = 0.8
        mock_sample_next_token = mocker.patch(
            "minigpt.sampling.sample_next_token",
            return_value=torch.randint(high=config.vocab_size, size=(1, 1)),
        )

        out = gpt_model.generate(
            idx,
            max_new_tokens,
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        assert out.shape == (1, input_length + max_new_tokens)
        assert mock_sample_next_token.call_count == max_new_tokens
        first_call = mock_sample_next_token.call_args_list[0]
        assert first_call == mocker.call(
            mocker.ANY, temperature=temperature, top_k=top_k, top_p=top_p
        )

    def test_generate_no_cache_context_growth(self, gpt_model):
        """Verifies non-cached generation feeds newly generated tokens into forward."""
        captured_inputs = []
        orig_forward = gpt_model.forward

        def spy_forward(idx, *args, **kwargs):
            captured_inputs.append(idx.clone())
            return orig_forward(idx, *args, **kwargs)

        gpt_model.forward = spy_forward

        prompt = torch.tensor([[1, 2, 3]])
        gpt_model.generate(prompt, max_new_tokens=3, use_cache=False)

        # Step 0: len == 3 (prompt)
        # Step 1: len == 4 (prompt + token 1)
        # Step 2: len == 5 (prompt + token 1 + token 2)
        for step, inp in enumerate(captured_inputs):
            expected_len = 3 + step
            assert inp.shape[1] == expected_len, (
                f"Step {step}: expected context length {expected_len}, "
                f"but model received static context of length {inp.shape[1]}!"
            )

    def test_generate_cache_parity_greedy(self, gpt_model):
        """Verifies that cached and non-cached generation produce identical tokens."""
        prompt_tokens = torch.tensor([[1, 2, 3]])
        max_new_tokens = 8

        out_cached = gpt_model.generate(
            prompt_tokens.clone(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.0,
        )
        out_no_cache = gpt_model.generate(
            prompt_tokens.clone(),
            max_new_tokens=max_new_tokens,
            use_cache=False,
            temperature=0.0,
        )

        torch.testing.assert_close(out_cached, out_no_cache)
