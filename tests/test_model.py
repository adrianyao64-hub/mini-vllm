"""M2 step 4: DecoderLayer -> Model -> ForCausalLM wired together. Property tests + HF comparisons."""
import pytest
import torch

from mini_vllm.config import Qwen3Config
from mini_vllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, RMSNorm
from mini_vllm.models.rope import RotaryEmbedding
from tests.test_loader import TINY  # vocab 32, hidden 16, 2 layers, 4 query heads, 2 KV heads, head_dim 8

hf_qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
from transformers import Qwen3Config as HFQwen3Config  # noqa: E402

CFG = Qwen3Config.from_dict(TINY)  # tie_word_embeddings=True, like Qwen3-0.6B


def randomize_norms(module: torch.nn.Module) -> None:
    """RMSNorm weights start at 1. Make them random so a skipped or swapped norm changes the output."""
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, RMSNorm):
                m.weight.copy_(torch.rand(m.weight.shape, generator=gen) + 0.5)


# ---------------------------------------------------------------- decoder layer

def make_layer(dtype=torch.float32) -> Qwen3DecoderLayer:
    torch.manual_seed(0)
    layer = Qwen3DecoderLayer(CFG)
    randomize_norms(layer)
    return layer.to(dtype).eval()


def layer_inputs(B=2, T=6, dtype=torch.float32):
    torch.manual_seed(1)
    x = torch.randn(B, T, CFG.hidden_size, dtype=dtype)
    positions = torch.arange(T).expand(B, T)
    cos, sin = RotaryEmbedding(CFG.head_dim, CFG.rope_theta)(positions, dtype)
    return x, cos, sin


def test_decoder_layer_is_identity_when_both_branches_output_zero():
    # o_proj = 0 and down_proj = 0 make both sub-layers output exactly 0, so a correct residual block
    # returns x unchanged. Catches x = norm(x) + attn(...) (normalizing the residual stream itself)
    # and a forgotten "x +".
    layer = make_layer()
    x, cos, sin = layer_inputs()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
        torch.testing.assert_close(layer(x, cos, sin), x, rtol=0, atol=0)


def hf_config() -> HFQwen3Config:
    return HFQwen3Config(
        vocab_size=CFG.vocab_size, hidden_size=CFG.hidden_size, intermediate_size=CFG.intermediate_size,
        num_hidden_layers=CFG.num_hidden_layers, num_attention_heads=CFG.num_attention_heads,
        num_key_value_heads=CFG.num_key_value_heads, head_dim=CFG.head_dim, rms_norm_eps=CFG.rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": CFG.rope_theta},
        tie_word_embeddings=CFG.tie_word_embeddings, attn_implementation="sdpa",
    )


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_decoder_layer_matches_hf(dtype, tol):
    ours = make_layer(dtype)
    hf = hf_qwen3.Qwen3DecoderLayer(hf_config(), layer_idx=0).to(dtype).eval()
    hf.load_state_dict(ours.state_dict())  # same names: input_layernorm, self_attn.*, post_attention_layernorm, mlp.*
    x, cos, sin = layer_inputs(dtype=dtype)
    with torch.no_grad():
        hf_out = hf(x, attention_mask=None, position_embeddings=(cos, sin))
        torch.testing.assert_close(ours(x, cos, sin), hf_out, rtol=tol, atol=tol)


# ---------------------------------------------------------------- whole model

def make_model(dtype=torch.float32) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(CFG)
    randomize_norms(model)
    return model.to(dtype).eval()  # .to() keeps the tie: it converts the shared Parameter in place


def token_ids(B=2, T=7):
    torch.manual_seed(1)
    return torch.randint(0, CFG.vocab_size, (B, T))


def test_logits_shape():
    model, ids = make_model(), token_ids()
    with torch.no_grad():
        assert model(ids).shape == (2, 7, CFG.vocab_size)


def test_tied_lm_head():
    # Score of token v = <final hidden state, embedding of v>.
    model, ids = make_model(), token_ids()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with torch.no_grad():
        hidden = model.model(ids)  # [B, T, hidden]
        torch.testing.assert_close(model(ids), hidden @ model.model.embed_tokens.weight.T)


def test_causal_whole_model():
    # Changing the last token must not change any earlier position's logits, through all layers.
    model, ids = make_model(), token_ids()
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % CFG.vocab_size
    with torch.no_grad():
        out, out2 = model(ids), model(ids2)
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])
    assert not torch.allclose(out[:, -1], out2[:, -1])


def test_default_positions_are_0_to_T_minus_1():
    model, ids = make_model(), token_ids()
    positions = torch.arange(ids.shape[1]).expand_as(ids)
    with torch.no_grad():
        torch.testing.assert_close(model(ids), model(ids, positions), rtol=0, atol=0)


def test_batch_rows_are_independent():
    # Row 0 computed alone == row 0 computed in a batch of 2. Catches view/reshape bugs that mix B and T.
    model, ids = make_model(), token_ids()
    with torch.no_grad():
        torch.testing.assert_close(model(ids)[:1], model(ids[:1]))


def test_rope_freqs_stay_fp32_after_casting_to_bf16():
    # Module.to(bf16) casts parameters AND buffers. inv_freq is a plain attribute (see rope.py), so it stays
    # FP32; as a buffer it would be rounded to BF16 and positions would drift.
    model = make_model(torch.bfloat16)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16


def hf_model(ours: Qwen3ForCausalLM, dtype) -> torch.nn.Module:
    hf = hf_qwen3.Qwen3ForCausalLM(hf_config()).eval()
    hf.load_state_dict(ours.state_dict())  # copy_ converts dtype; built in FP32 first, cast below
    # HF keeps inv_freq as a buffer, so hf.to(bf16) would round HF's own RoPE frequencies to BF16
    # (from_pretrained keeps them FP32). Cast everything EXCEPT rotary_emb.
    for name, child in hf.model.named_children():
        if name != "rotary_emb":
            child.to(dtype)
    hf.lm_head.to(dtype)
    return hf


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
def test_full_model_matches_hf(dtype, tol):
    # Random 2-layer model, same weights in ours and HF's Qwen3ForCausalLM; compare logits at every position.
    # BF16 tolerance is loose: errors pile up over 2 layers. The real proof is step 5 (identical greedy text).
    ours = make_model(dtype)
    hf = hf_model(ours, dtype)
    ids = token_ids()
    positions = torch.arange(ids.shape[1]).expand_as(ids)
    with torch.no_grad():
        hf_logits = hf(input_ids=ids, position_ids=positions).logits
        torch.testing.assert_close(ours(ids, positions), hf_logits, rtol=tol, atol=tol)
