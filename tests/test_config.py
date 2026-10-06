import json
from pathlib import Path

import pytest

from mini_vllm.config import Qwen3Config

# Copied from Qwen/Qwen3-0.6B config.json (trimmed), so tests don't need the checkpoint.
QWEN3_0_6B = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}


def write_config(tmp_path: Path, cfg: dict) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_qwen3_0_6b(tmp_path):
    cfg = Qwen3Config.from_pretrained(write_config(tmp_path, QWEN3_0_6B))
    assert cfg.head_dim == 128                  # explicit, not 1024 // 16 = 64
    assert cfg.num_kv_groups == 2               # GQA: 16 query heads / 8 KV heads
    assert cfg.q_size == 2048                   # matches q_proj.weight [2048, 1024]
    assert cfg.kv_size == 1024                  # matches k_proj.weight [1024, 1024]
    assert cfg.tie_word_embeddings is True


def test_unknown_keys_ignored():
    cfg = Qwen3Config.from_dict({**QWEN3_0_6B, "some_new_field": 123})
    assert not hasattr(cfg, "some_new_field")


def test_missing_head_dim_raises():
    # head_dim must come from config.json; deriving 1024 // 16 = 64 would silently be wrong for Qwen3.
    d = {k: v for k, v in QWEN3_0_6B.items() if k != "head_dim"}
    with pytest.raises(ValueError, match="head_dim"):
        Qwen3Config.from_dict(d)


def test_rejects_unsupported():
    with pytest.raises(ValueError):
        Qwen3Config.from_dict({**QWEN3_0_6B, "model_type": "llama"})
    with pytest.raises(ValueError):
        Qwen3Config.from_dict({**QWEN3_0_6B, "model_type": "qwen2"})
    with pytest.raises(NotImplementedError):
        Qwen3Config.from_dict({**QWEN3_0_6B, "attention_bias": True})
    with pytest.raises(ValueError):
        Qwen3Config.from_dict({**QWEN3_0_6B, "num_key_value_heads": 3})
    with pytest.raises(NotImplementedError):
        Qwen3Config.from_dict({**QWEN3_0_6B, "rope_scaling": {"type": "yarn"}})


CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-0.6B"


@pytest.mark.skipif(not (CKPT / "config.json").exists(), reason="checkpoint not downloaded")
def test_real_checkpoint_config():
    cfg = Qwen3Config.from_pretrained(CKPT)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.head_dim) == (28, 1024, 128)
