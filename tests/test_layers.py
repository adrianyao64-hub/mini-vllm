"""M2 step 1: RMSNorm and SwiGLU MLP, checked against a hand-written formula and against HF's modules."""
import pytest
import torch
import torch.nn.functional as F

from mini_vllm.config import Qwen3Config
from mini_vllm.models.qwen3 import Qwen3MLP, RMSNorm
from tests.test_loader import TINY

hf_qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
from transformers import Qwen3Config as HFQwen3Config  # noqa: E402


def make_norm(dim: int, eps: float = 1e-6) -> RMSNorm:
    torch.manual_seed(0)
    norm = RMSNorm(dim, eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim))  # random weight so "weight == 1" can't hide bugs
    return norm


def test_rmsnorm_formula():
    norm = make_norm(16)
    x = torch.randn(2, 3, 16)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_rmsnorm_unit_rms():
    # With weight = 1, every output vector has root-mean-square 1 -- that's what "RMS norm" means.
    norm = RMSNorm(16, 1e-6)
    y = norm(torch.randn(4, 16) * 50)
    torch.testing.assert_close(y.pow(2).mean(-1).sqrt(), torch.ones(4))


def test_rmsnorm_keeps_dtype():
    norm = make_norm(16).to(torch.bfloat16)
    assert norm(torch.randn(2, 16, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_rmsnorm_per_head():
    # q_norm is applied to [B, T, heads, head_dim]: each head is normalized on its own.
    norm = make_norm(8)
    x = torch.randn(2, 3, 4, 8)
    torch.testing.assert_close(norm(x)[:, :, 1], norm(x[:, :, 1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype):
    ours = make_norm(16).to(dtype)
    hf = hf_qwen3.Qwen3RMSNorm(16, eps=1e-6).to(dtype)
    hf.load_state_dict(ours.state_dict())
    x = torch.randn(2, 5, 16, dtype=dtype) * 10
    # Same ops in the same order -> should be bit-identical, not just "close".
    torch.testing.assert_close(ours(x), hf(x), rtol=0, atol=0)


def test_mlp_formula():
    torch.manual_seed(0)
    mlp = Qwen3MLP(Qwen3Config.from_dict(TINY))
    x = torch.randn(2, 3, TINY["hidden_size"])
    expected = (F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T
    torch.testing.assert_close(mlp(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_matches_hf(dtype):
    torch.manual_seed(0)
    ours = Qwen3MLP(Qwen3Config.from_dict(TINY)).to(dtype)
    hf_cfg = HFQwen3Config(hidden_size=TINY["hidden_size"], intermediate_size=TINY["intermediate_size"],
                           hidden_act="silu")
    hf = hf_qwen3.Qwen3MLP(hf_cfg).to(dtype)
    hf.load_state_dict(ours.state_dict())  # same attribute names -> load directly, no renaming
    x = torch.randn(2, 5, TINY["hidden_size"], dtype=dtype)
    torch.testing.assert_close(ours(x), hf(x), rtol=0, atol=0)
