"""M2 step 2: RoPE -- math properties, and bit-exact match with HF's Qwen3 RoPE."""
import pytest
import torch

from mini_vllm.models.rope import RotaryEmbedding, apply_rope, rotate_half

hf_qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
from transformers import Qwen3Config as HFQwen3Config  # noqa: E402

THETA = 1_000_000.0  # Qwen3 rope_theta
D = 128              # Qwen3 head_dim


def rope_fp32(positions, d=D):
    return RotaryEmbedding(d, THETA)(positions, torch.float32)


def test_inv_freq():
    inv = RotaryEmbedding(D, THETA).inv_freq
    assert inv.shape == (D // 2,)
    assert inv[0] == 1.0                                      # pair 0: 1 rad per token
    torch.testing.assert_close(inv[-1], torch.tensor(THETA ** (-126 / 128)))  # pair 63: ~1.2e-6
    assert torch.all(inv[1:] < inv[:-1])                      # strictly slower and slower


def test_no_state_and_works_after_meta():
    # Built under meta like loader.py does: inv_freq must still be a real CPU tensor,
    # and there must be nothing for to_empty() to wipe or for the loader to look for.
    with torch.device("meta"):
        rope = RotaryEmbedding(D, THETA)
    assert rope.inv_freq.device.type == "cpu"
    assert list(rope.parameters()) == [] and list(rope.buffers()) == []
    cos, sin = rope(torch.tensor([[3]]), torch.float32)
    torch.testing.assert_close(cos, rope_fp32(torch.tensor([[3]]))[0])


def test_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_pairing_is_front_half_with_back_half():
    # head_dim 4 -> pairs (dim0, dim2) and (dim1, dim3); check the 2-D rotation formula by hand.
    pos = torch.tensor([[5]])
    cos, sin = RotaryEmbedding(4, THETA)(pos, torch.float32)
    a, b, c, d = 1.0, 2.0, 3.0, 4.0
    x = torch.tensor([a, b, c, d]).view(1, 1, 1, 4)
    out = apply_rope(x, cos, sin).flatten()
    t0, t1 = 5 * 1.0, 5 * THETA ** (-2 / 4)  # angles of pair 0 and pair 1
    expected = torch.tensor([
        a * torch.cos(torch.tensor(t0)) - c * torch.sin(torch.tensor(t0)),
        b * torch.cos(torch.tensor(t1)) - d * torch.sin(torch.tensor(t1)),
        c * torch.cos(torch.tensor(t0)) + a * torch.sin(torch.tensor(t0)),
        d * torch.cos(torch.tensor(t1)) + b * torch.sin(torch.tensor(t1)),
    ])
    torch.testing.assert_close(out, expected)


def test_position_zero_is_identity():
    x = torch.randn(1, 1, 4, D)
    cos, sin = rope_fp32(torch.zeros(1, 1, dtype=torch.long))
    torch.testing.assert_close(apply_rope(x, cos, sin), x)  # angle 0: cos = 1, sin = 0


def test_preserves_length():
    x = torch.randn(2, 6, 4, D)
    cos, sin = rope_fp32(torch.randint(0, 40960, (2, 6)))
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_dot_product_depends_only_on_relative_position():
    # The reason RoPE exists: q at position m and k at position n score the same as at m+s, n+s.
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)

    def score(m, n):
        cq, sq = rope_fp32(torch.tensor([[m]]))
        ck, sk = rope_fp32(torch.tensor([[n]]))
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()

    torch.testing.assert_close(score(5, 2), score(105, 102), rtol=1e-3, atol=1e-3)
    # Looser far away: an FP32 angle near 1000 rad is only accurate to ~6e-5 rad, and the
    # error grows with position. (In BF16 the angle itself would be off by whole radians.)
    torch.testing.assert_close(score(5, 2), score(1005, 1002), rtol=1e-3, atol=1e-2)


def hf_rope():
    cfg = HFQwen3Config(hidden_size=1024, num_attention_heads=16, head_dim=D, max_position_embeddings=40960,
                        rope_parameters={"rope_type": "default", "rope_theta": THETA})
    return hf_qwen3.Qwen3RotaryEmbedding(cfg)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cos_sin_match_hf(dtype):
    positions = torch.randint(0, 40960, (2, 7))
    cos, sin = RotaryEmbedding(D, THETA)(positions, dtype)
    hf_cos, hf_sin = hf_rope()(torch.empty(0, dtype=dtype), positions)  # HF only reads x's dtype / device
    torch.testing.assert_close(cos, hf_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, hf_sin, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_matches_hf(dtype):
    positions = torch.randint(0, 40960, (2, 7))
    q = torch.randn(2, 7, 16, D, dtype=dtype)  # ours: [B, T, heads, D]
    k = torch.randn(2, 7, 8, D, dtype=dtype)
    cos, sin = RotaryEmbedding(D, THETA)(positions, dtype)
    # HF works on [B, heads, T, D] and unsqueezes cos/sin at dim 1; we transpose in and out.
    hf_q, hf_k = hf_qwen3.apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin, unsqueeze_dim=1)
    torch.testing.assert_close(apply_rope(q, cos, sin), hf_q.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(apply_rope(k, cos, sin), hf_k.transpose(1, 2), rtol=0, atol=0)
