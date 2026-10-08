"""M2 step 3: Qwen3Attention -- against a step-by-step reference and against HF's Qwen3Attention."""
import math

import pytest
import torch

from mini_vllm.config import Qwen3Config
from mini_vllm.models.qwen3 import Qwen3Attention
from mini_vllm.models.rope import RotaryEmbedding, apply_rope
from tests.test_loader import TINY  # hidden 16, 4 query heads, 2 KV heads, head_dim 8

hf_qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
from transformers import Qwen3Config as HFQwen3Config  # noqa: E402

CFG = Qwen3Config.from_dict(TINY)


def make_attn(dtype=torch.float32) -> Qwen3Attention:
    torch.manual_seed(0)
    attn = Qwen3Attention(CFG)
    with torch.no_grad():  # random norm weights too, so a skipped q_norm / k_norm can't hide
        attn.q_norm.weight.copy_(torch.rand(CFG.head_dim) + 0.5)
        attn.k_norm.weight.copy_(torch.rand(CFG.head_dim) + 0.5)
    return attn.to(dtype)


def inputs(B=2, T=6, dtype=torch.float32):
    torch.manual_seed(1)
    x = torch.randn(B, T, CFG.hidden_size, dtype=dtype)
    positions = torch.arange(T).expand(B, T)
    cos, sin = RotaryEmbedding(CFG.head_dim, CFG.rope_theta)(positions, dtype)
    return x, positions, cos, sin


def reference_attention(attn: Qwen3Attention, x, cos, sin):
    """The same computation written out one operation at a time, with GQA done by explicit copying."""
    B, T, _ = x.shape
    H, KV, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = apply_rope(attn.q_norm(attn.q_proj(x).view(B, T, H, D)), cos, sin).transpose(1, 2)   # [B, H, T, D]
    k = apply_rope(attn.k_norm(attn.k_proj(x).view(B, T, KV, D)), cos, sin).transpose(1, 2)  # [B, KV, T, D]
    v = attn.v_proj(x).view(B, T, KV, D).transpose(1, 2)
    # GQA: KV head j serves query heads 2j and 2j+1 -> repeat_interleave gives [k0, k0, k1, k1].
    # (k.repeat(1, 2, 1, 1) would give [k0, k1, k0, k1]: wrong heads, no error.)
    k = k.repeat_interleave(H // KV, dim=1)
    v = v.repeat_interleave(H // KV, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(D)                       # [B, H, T, T]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # True above the diagonal
    scores = scores.masked_fill(future, float("-inf"))                    # exp(-inf) = 0 after softmax
    probs = scores.softmax(dim=-1)                                        # each row sums to 1
    out = (probs @ v).transpose(1, 2).reshape(B, T, H * D)
    return attn.o_proj(out)


def test_output_shape():
    attn = make_attn()
    x, _, cos, sin = inputs()
    assert attn(x, cos, sin).shape == x.shape


def test_matches_step_by_step_reference():
    attn = make_attn()
    x, _, cos, sin = inputs()
    with torch.no_grad():
        torch.testing.assert_close(attn(x, cos, sin), reference_attention(attn, x, cos, sin))


def test_causal_future_tokens_do_not_matter():
    # Changing the last token must not change the outputs of any earlier token.
    attn = make_attn()
    x, _, cos, sin = inputs()
    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1]) * 10
    with torch.no_grad():
        out, out2 = attn(x, cos, sin), attn(x2, cos, sin)
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])
    assert not torch.allclose(out[:, -1], out2[:, -1])


def test_first_token_only_sees_itself():
    # Token 0 attends only to token 0: softmax over one score is 1, so the output is o_proj(v_0),
    # whatever q and k are.
    attn = make_attn()
    x, _, cos, sin = inputs()
    with torch.no_grad():
        v0 = attn.v_proj(x[:, :1]).view(2, 1, CFG.num_key_value_heads, CFG.head_dim)
        v0 = v0.repeat_interleave(CFG.num_kv_groups, dim=2).reshape(2, 1, -1)
        torch.testing.assert_close(attn(x, cos, sin)[:, :1], attn.o_proj(v0))


def hf_attention(ours: Qwen3Attention, dtype) -> torch.nn.Module:
    cfg = HFQwen3Config(
        hidden_size=CFG.hidden_size, num_attention_heads=CFG.num_attention_heads,
        num_key_value_heads=CFG.num_key_value_heads, head_dim=CFG.head_dim, rms_norm_eps=CFG.rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": CFG.rope_theta}, attn_implementation="sdpa",
    )
    hf = hf_qwen3.Qwen3Attention(cfg, layer_idx=0).to(dtype).eval()
    hf.load_state_dict(ours.state_dict())  # same names: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
    return hf


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_matches_hf(dtype, tol):
    # Not bit-exact any more: attention sums over many tokens, and different kernels add in different
    # orders. So from here on we compare with a tolerance (the end-to-end test checks generated text).
    ours = make_attn(dtype)
    hf = hf_attention(ours, dtype)
    x, _, cos, sin = inputs(dtype=dtype)
    with torch.no_grad():
        hf_out, _ = hf(x, (cos, sin), attention_mask=None)  # mask=None -> HF's SDPA path uses is_causal=True
        torch.testing.assert_close(ours(x, cos, sin), hf_out, rtol=tol, atol=tol)
