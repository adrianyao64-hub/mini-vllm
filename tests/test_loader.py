from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mini_vllm.config import Qwen3Config
from mini_vllm.loader import load_model, load_weights
from mini_vllm.models.qwen3 import Qwen3ForCausalLM
from tests.test_config import QWEN3_0_6B

# A tiny Qwen3 (2 layers, hidden 16) so tests run in milliseconds without the real checkpoint.
TINY = {**QWEN3_0_6B, "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8}


def make_tiny_checkpoint(tmp_path: Path, tie: bool = True) -> dict[str, torch.Tensor]:
    """Write a random tiny checkpoint laid out like the real one (including the redundant lm_head)."""
    import json
    cfg = {**TINY, "tie_word_embeddings": tie}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    ref = Qwen3ForCausalLM(Qwen3Config.from_dict(cfg))
    # safetensors refuses tensors that share memory, so store lm_head as its own copy like HF does.
    weights = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    save_file(weights, str(tmp_path / "model.safetensors"))
    return weights


def test_load_tiny_roundtrip(tmp_path):
    weights = make_tiny_checkpoint(tmp_path)
    model = load_model(tmp_path, dtype=torch.float32)
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name]), name
    assert model.lm_head.weight is model.model.embed_tokens.weight  # tie survived to_empty()


def test_missing_tensor_raises(tmp_path):
    weights = make_tiny_checkpoint(tmp_path)
    del weights["model.layers.1.mlp.up_proj.weight"]
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        load_model(tmp_path)


def test_shape_mismatch_raises(tmp_path):
    weights = make_tiny_checkpoint(tmp_path)
    weights["model.norm.weight"] = torch.ones(17)
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_model(tmp_path)


CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-0.6B"


@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="checkpoint not downloaded")
def test_load_real_checkpoint():
    model = load_model(CKPT)  # bf16 on CPU, ~1.2 GB
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with safe_open(CKPT / "model.safetensors", framework="pt") as f:
        for name in ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                     "model.layers.27.mlp.down_proj.weight", "model.norm.weight"]:
            assert torch.equal(model.get_parameter(name), f.get_tensor(name)), name
        # The redundant lm_head in the file is bit-identical to the embedding, so sharing is safe.
        assert torch.equal(f.get_tensor("lm_head.weight"), f.get_tensor("model.embed_tokens.weight"))
