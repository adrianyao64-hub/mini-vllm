"""Qwen3 model definition (decoder-only Transformer).

M1: module structure and parameter shapes only, named exactly like the HF checkpoint
    (e.g. `model.layers.0.self_attn.q_proj.weight`) so weights load 1:1 with no renaming.
M2: forward passes (RMSNorm, RoPE, attention, SwiGLU) matched against HF logits.

Shapes below are for Qwen3-0.6B: hidden=1024, 16 query heads, 8 KV heads, head_dim=128.
"""
from __future__ import annotations

import torch
from torch import nn

from mini_vllm.config import Qwen3Config


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. Only a scale vector, no bias."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")


class Qwen3Attention(nn.Module):
    """Grouped-query attention with per-head RMSNorm on q and k (Qwen3's QK-Norm)."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads       # 16
        self.num_kv_heads = config.num_key_value_heads    # 8
        self.head_dim = config.head_dim                   # 128

        # nn.Linear(in, out) stores weight as [out, in]. Qwen3 has no bias on any projection.
        self.q_proj = nn.Linear(config.hidden_size, config.q_size, bias=False)   # [2048, 1024]
        self.k_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)  # [1024, 1024]
        self.v_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)  # [1024, 1024]
        self.o_proj = nn.Linear(config.q_size, config.hidden_size, bias=False)   # [1024, 2048]

        # Normalizes each head's 128-dim q / k vector, so the size is head_dim, not hidden_size.
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # [128]
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # [128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # [3072, 1024]
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # [3072, 1024]
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # [1024, 3072]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")


class Qwen3Model(nn.Module):
    """Embedding -> N decoder layers -> final norm. Outputs hidden states, not logits."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # [151936, 1024]
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")


class Qwen3ForCausalLM(nn.Module):
    """Qwen3Model + lm_head that turns hidden states into vocab logits."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)  # attribute name "model" -> parameter prefix "model."
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # [151936, 1024]
        self.tie_weights()

    def tie_weights(self) -> None:
        """Point lm_head at the SAME tensor as the embedding: one copy in memory, saves ~300 MB.

        Called again after moving the model off the meta device, because to_empty() can break the tie.
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("M2")
