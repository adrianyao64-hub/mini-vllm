"""M2 step 6: our Qwen3 vs HF `from_pretrained` on the REAL Qwen3-0.6B weights (CPU).

Marked `slow` (loads two 0.6B models per dtype and decodes without a KV cache, ~1-3 min on CPU), so the
default `uv run pytest` deselects it (see addopts in pyproject.toml). Run it with:

    uv run pytest -m slow -v

Skipped when checkpoints/Qwen3-0.6B is not downloaded (e.g. in CI).
The logic is shared with scripts/compare_hf.py, which prints the numbers for the milestone note.

What is enforced and why:
- Greedy text must match HF token by token: this is the real correctness gate.
- Logits only need argmax agreement + a loose tolerance. On CPU they are currently bit-identical (error 0),
  but that depends on HF internals; a transformers upgrade could add tiny rounding noise and a bitwise
  assert would then fail for no real bug.
- hf.generate (KV cache) is only compared in FP32: in BF16 exact logit ties exist, and the cached decode
  rounds differently, so a tie may flip (see the 2026-10-08 entry in the M2 milestone note).
"""
from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

from mini_vllm.loader import load_model  # noqa: E402
from scripts.compare_hf import PROMPTS, compare_logits, greedy_no_cache  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-0.6B"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="checkpoint not downloaded"),
]

MAX_NEW_TOKENS = 32
# Loose on purpose (measured: 0 in both dtypes). BF16 logits near 20 are spaced 0.125 apart.
LOGITS_TOL = {torch.float32: 1e-3, torch.bfloat16: 1e-1}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CKPT)


@pytest.fixture(scope="module")
def gen_defaults() -> GenerationConfig:
    return GenerationConfig.from_pretrained(CKPT)  # eos [151645, 151643], pad 151643


@pytest.fixture(scope="module")
def eos_ids(gen_defaults) -> set[int]:
    eos = gen_defaults.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def models(request):
    """(dtype, ours, hf) for one dtype. Module scope: pytest groups tests by dtype, so each pair loads once."""
    dtype = request.param
    ours = load_model(CKPT, device="cpu", dtype=dtype)
    hf = AutoModelForCausalLM.from_pretrained(CKPT, dtype=dtype, attn_implementation="sdpa").eval()
    yield dtype, ours, hf
    del ours, hf
    gc.collect()  # free ~2.4 GB (FP32) before the next dtype loads


def prompt_ids(tok, prompt: str) -> torch.Tensor:
    return tok(prompt, return_tensors="pt").input_ids  # [1, T], no padding


@torch.no_grad()
def test_real_logits_match_hf(models, tok):
    dtype, ours, hf = models
    for prompt in PROMPTS:
        ids = prompt_ids(tok, prompt)
        r = compare_logits(ours(ids), hf(input_ids=ids, use_cache=False).logits)
        assert r.argmax_agree == 1.0, (prompt, r)
        assert r.top5_same, (prompt, r)
        assert r.max_abs <= LOGITS_TOL[dtype], (prompt, r)


def test_greedy_matches_hf(models, tok, eos_ids):
    # Both sides run the SAME no-cache loop, so only the models differ.
    dtype, ours, hf = models
    for prompt in PROMPTS:
        ids = prompt_ids(tok, prompt)
        g_ours = greedy_no_cache(lambda x: ours(x), ids, MAX_NEW_TOKENS, eos_ids)
        g_hf = greedy_no_cache(lambda x: hf(input_ids=x, use_cache=False).logits, ids, MAX_NEW_TOKENS, eos_ids)
        assert g_ours.tokens == g_hf.tokens, (
            f"{prompt!r}\n  ours: {tok.decode(g_ours.tokens)!r}\n  hf:   {tok.decode(g_hf.tokens)!r}"
        )


@pytest.mark.parametrize("models", [torch.float32], indirect=True, ids=["fp32"])
def test_greedy_matches_hf_generate_fp32(models, tok, eos_ids, gen_defaults):
    # HF's own generate() uses a KV cache: the reference M3 will be compared against.
    _, ours, hf = models
    cfg = GenerationConfig(  # fresh greedy config: the checkpoint's default is do_sample=True
        do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=sorted(eos_ids), pad_token_id=gen_defaults.pad_token_id,
    )
    for prompt in PROMPTS:
        ids = prompt_ids(tok, prompt)
        g_ours = greedy_no_cache(lambda x: ours(x), ids, MAX_NEW_TOKENS, eos_ids)
        with torch.no_grad():
            gen = hf.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
        gen_tokens = gen[0, ids.shape[1]:].tolist()
        assert g_ours.tokens == gen_tokens, (
            f"{prompt!r}\n  ours:         {tok.decode(g_ours.tokens)!r}\n  hf.generate:  {tok.decode(gen_tokens)!r}"
        )
