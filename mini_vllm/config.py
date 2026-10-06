"""Model config parsed from a Hugging Face `config.json`.

Only the fields the engine actually uses are kept; everything else in config.json is ignored.
Supports Qwen3 only. Qwen3.5 (hybrid linear/full attention) gets its own config class in M7.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

REQUIRED_KEYS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
)


@dataclass(frozen=True)
class Qwen3Config:
    # --- sizes ---
    vocab_size: int
    hidden_size: int
    intermediate_size: int          # MLP hidden size (gate/up output dim)
    num_hidden_layers: int
    num_attention_heads: int        # query heads
    num_key_value_heads: int        # K/V heads (< query heads => GQA)
    head_dim: int                   # explicit in Qwen3 (128); NOT hidden_size // heads (64)
    # --- numerics / position ---
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    torch_dtype: str = "bfloat16"
    tie_word_embeddings: bool = False  # lm_head shares embed_tokens weight
    # --- generation ---
    eos_token_id: int | list[int] | None = None

    def __post_init__(self) -> None:
        # Fail early on configs the model code cannot handle.
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share one K/V head (2 for Qwen3-0.6B)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_size(self) -> int:
        """Output dim of q_proj = query heads * head_dim (2048 for Qwen3-0.6B)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """Output dim of k_proj / v_proj = K/V heads * head_dim (1024 for Qwen3-0.6B)."""
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3Config:
        if d.get("model_type") != "qwen3":
            raise ValueError(f"unsupported model_type {d.get('model_type')!r}, expected 'qwen3'")
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"config.json is missing required keys: {missing}")
        if d.get("attention_bias"):
            raise NotImplementedError("attention_bias=True is not supported (Qwen3 has no QKV bias)")
        if d.get("rope_scaling") is not None:
            raise NotImplementedError(f"rope_scaling is not supported yet: {d['rope_scaling']}")

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3Config:
        with open(Path(model_dir) / "config.json", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
