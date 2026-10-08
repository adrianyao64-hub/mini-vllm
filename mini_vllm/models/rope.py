"""Rotary position embedding (RoPE), written to match HF's Qwen3 implementation bit for bit.

Layout in mini-vllm: q / k are [B, T, heads, head_dim], and positions are [B, T] integers.

Idea: split each head's 128-dim vector into 64 2-D pairs and rotate pair i by the angle
    position * inv_freq[i].
Rotation keeps the vector's length, and the dot product of two rotated vectors only depends
on the DIFFERENCE of their positions -> attention scores see relative position.
"""
from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """positions [B, T] -> (cos, sin) tables [B, T, head_dim] that apply_rope() multiplies q / k by.

    Has no parameters and no buffers. inv_freq (64 numbers) is a plain tensor computed on CPU:
    - a registered buffer would be wiped by loader.py's meta -> to_empty() path (it isn't in the
      checkpoint, so nothing would refill it);
    - device="cpu" is explicit so it is real even when the model is built under torch.device("meta"),
      and computing on CPU (like HF does) gives exactly HF's bits.
    """

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        # Pair i rotates at speed theta^(-2i/d) radians per position:
        #   i = 0  -> 1 rad / token        (fast: distinguishes nearby tokens)
        #   i = 63 -> ~1.2e-6 rad / token  (slow: still meaningful after tens of thousands of tokens)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float, device="cpu") / head_dim  # [64]
        self.inv_freq = 1.0 / (float(theta) ** exponent)  # same expression as HF -> same bits

    @torch.no_grad()
    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [B, T] ints. Returns cos, sin: [B, T, head_dim], cast to `dtype` (like HF)."""
        inv_freq = self.inv_freq.to(positions.device)
        # Angles in FP32: position (up to 40960) * frequency. BF16 would round 40000 to a multiple of 256.
        freqs = positions.float()[..., None] * inv_freq   # [B, T, 64]: angle of pair i at each position
        emb = torch.cat((freqs, freqs), dim=-1)            # [B, T, 128]: dims i and i+64 form pair i
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], where x1 / x2 are the FRONT / BACK half of the last dim.

    HF pairs dim i with dim i + head_dim/2 (not with i+1 like the original RoPE paper).
    The checkpoint was trained with this pairing, so we must use the same one.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k. x: [B, T, heads, head_dim]; cos / sin: [B, T, head_dim].

    For each pair (a, b) = (x[i], x[i+64]) with angle t:
        a' = a*cos t - b*sin t
        b' = b*cos t + a*sin t        <- a 2-D rotation by t
    Written as x*cos + rotate_half(x)*sin, which does all 64 pairs at once with no Python loop.
    """
    cos = cos.unsqueeze(-2)  # [B, T, 1, head_dim]: broadcast the same angles to every head
    sin = sin.unsqueeze(-2)
    return x * cos + rotate_half(x) * sin
