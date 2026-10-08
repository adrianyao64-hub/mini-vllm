"""Qwen3 model definition (decoder-only Transformer).

M1: module structure and parameter shapes only, named exactly like the HF checkpoint
    (e.g. `model.layers.0.self_attn.q_proj.weight`) so weights load 1:1 with no renaming.
M2: forward passes (RMSNorm, RoPE, attention, SwiGLU) matched against HF logits.
    Interface: model(input_ids [B, T], positions [B, T]) -> logits [B, T, vocab]. No KV cache yet (M3).

Shapes below are for Qwen3-0.6B: hidden=1024, 16 query heads, 8 KV heads, head_dim=128.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_vllm.config import Qwen3Config
from mini_vllm.models.rope import RotaryEmbedding, apply_rope


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. Only a scale vector, no bias."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]. Normalizes over the last dim only, so the same module works for
        # hidden states [B, T, 1024] and for per-head q/k vectors [B, T, heads, 128].
        input_dtype = x.dtype
        # Square-and-mean in BF16 loses precision (8-bit mantissa), so compute in FP32 like HF.
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)  # [..., 1]
        x = x * torch.rsqrt(variance + self.eps)         # rsqrt = 1 / sqrt
        # Same order as HF: cast back to the input dtype first, THEN multiply by weight.
        # (Multiplying in FP32 and casting afterwards rounds differently -> logits drift from HF.)
        return self.weight * x.to(input_dtype)


class Qwen3Attention(nn.Module):
    """Grouped-query attention with per-head RMSNorm on q and k (Qwen3's QK-Norm)."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads       # 16
        self.num_kv_heads = config.num_key_value_heads    # 8
        self.head_dim = config.head_dim                   # 128
        self.num_kv_groups = config.num_kv_groups         # 2 query heads share each K/V head
        self.scale = self.head_dim ** -0.5                # 1/sqrt(128): keeps q.k scores from growing with dim

        # nn.Linear(in, out) stores weight as [out, in]. Qwen3 has no bias on any projection.
        self.q_proj = nn.Linear(config.hidden_size, config.q_size, bias=False)   # [2048, 1024]
        self.k_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)  # [1024, 1024]
        self.v_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)  # [1024, 1024]
        self.o_proj = nn.Linear(config.q_size, config.hidden_size, bias=False)   # [1024, 2048]

        # Normalizes each head's 128-dim q / k vector, so the size is head_dim, not hidden_size.
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # [128]
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # [128]

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden]; cos / sin: [B, T, head_dim] from RotaryEmbedding. Returns [B, T, hidden].

        M2 limits: no KV cache (M3), and every sequence in the batch has the same length with no
        padding (the causal mask below assumes token t can see tokens 0..t of its own row).
        """
        B, T, _ = x.shape
        # 1) Project and split into heads. view() is free: it only reinterprets the last dim 2048 as 16 x 128.
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)     # [B, T, 16, 128]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)  # [B, T,  8, 128]
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)  # [B, T,  8, 128]

        # 2) QK-Norm, THEN RoPE (same order as HF). RoPE only rotates, so norm-then-rotate keeps the
        #    rotation intact; rotating first and normalizing after would scale the rotated vector instead.
        #    v gets neither: position should change WHO a token attends to, not WHAT it reads.
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        # 3) SDPA wants heads before time: [B, heads, T, 128].
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        # 4) softmax(q k^T * scale + causal_mask) v  for every head.
        #    is_causal=True: token t only sees tokens <= t (a decoder must not peek at the future).
        #    enable_gqa=True: query head h reads K/V head h // 2 without copying K/V up to 16 heads.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)

        # 5) Merge heads back ([B, 16, T, 128] -> [B, T, 2048]) and mix them with o_proj.
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # [3072, 1024]
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # [3072, 1024]
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # [1024, 3072]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 1024] -> gate / up: [..., 3072] -> down: [..., 1024]
        # gate decides "how much of each feature passes" (via SiLU), up carries the content.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden]; cos / sin: [B, T, head_dim], shared by all layers. Returns [B, T, hidden].

        Pre-norm: each sub-layer reads a NORMALIZED copy of x, and its output is ADDED back to x.
        The residual stream x itself is never normalized, so the embedding signal flows through all
        28 layers untouched and every layer only adds a correction on top of it.
        """
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)  # tokens exchange information
        x = x + self.mlp(self.post_attention_layernorm(x))         # each token processes itself
        return x


class Qwen3Model(nn.Module):
    """Embedding -> N decoder layers -> final norm. Outputs hidden states, not logits."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # [151936, 1024]
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # One RoPE table builder for the whole model. It has no parameters and no buffers (see rope.py),
        # so the state_dict still holds exactly the checkpoint's 311 tensors.
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids: [B, T] token ids; positions: [B, T] position of each token (default 0..T-1).

        Returns the final hidden states [B, T, hidden] (after the last norm), not logits.
        positions is explicit because later it is NOT 0..T-1: in M3's decode step T = 1 and the
        position is the current sequence length; in M4 tokens of many requests are packed together.
        """
        if positions is None:
            B, T = input_ids.shape
            positions = torch.arange(T, device=input_ids.device).expand(B, T)
        x = self.embed_tokens(input_ids)                # [B, T, 1024], in the weights' dtype (BF16)
        # cos / sin depend only on the positions, not on the layer: compute once, reuse in all 28 layers.
        cos, sin = self.rotary_emb(positions, x.dtype)  # [B, T, 128] each
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


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

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids / positions: [B, T]. Returns logits [B, T, vocab]: a score for every vocab entry
        at every position. logits[:, -1] scores the token that comes AFTER the input (what generation uses).
        """
        hidden = self.model(input_ids, positions)  # [B, T, 1024]
        # Tied: lm_head.weight IS embed_tokens.weight [151936, 1024], so the score of token v is the dot
        # product of the hidden state with v's own embedding. Kept in the model dtype (BF16), like HF.
        # M2 scores all T positions (handy for comparing with HF); M3 will only score the last one.
        return self.lm_head(hidden)                # [B, T, 151936]
