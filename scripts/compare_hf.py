"""M2 step 5: end-to-end check of our Qwen3 forward against Hugging Face on the REAL Qwen3-0.6B weights.

Two checks per dtype:
  1) Logits: same prompt through both models, compare logits at every position
     (max / mean abs error, per-position argmax agreement, last-position top-5).
  2) Greedy decoding, up to 32 new tokens:
     - main check:  both models run the SAME no-KV-cache loop (re-run the whole sequence every step,
                    take argmax of the last position), so only the models differ, not the algorithm;
     - reference:   HF's own hf.generate(do_sample=False), which uses a KV cache. Reported, not enforced:
                    cached decode sums in a different order, so a BF16 near-tie may flip there.
  On a mismatch it prints where the two sequences split and the top-1 / top-2 logit gap at that step:
  a tiny gap means a rounding near-tie, a large gap means a real bug.

Exit code is 1 if any main-check greedy sequence differs from HF's, so this can back a slow test (step 6).

Usage (from the repo root):
    uv run python scripts/compare_hf.py                        # CPU, FP32 and BF16
    uv run python scripts/compare_hf.py --dtype bf16 --device mps
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# mini-vllm is a uv "virtual" project (not installed into .venv), and `python scripts/x.py` puts scripts/
# on sys.path, not the repo root. Add the root so `import mini_vllm` works (pytest does this via pythonpath).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mini_vllm.loader import load_model  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "Qwen3-0.6B"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# Different lengths and scripts (English / Chinese / code) so tokenization and positions vary.
PROMPTS = [
    "The capital of France is",
    "大语言模型推理的两个阶段分别是",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


@dataclass
class LogitsResult:
    max_abs: float
    mean_abs: float
    argmax_agree: float   # fraction of positions where both models pick the same top token
    top5_same: bool       # last position: same 5 tokens in the same order


@dataclass
class GreedyResult:
    tokens: list[int]                  # generated tokens only (prompt excluded)
    top2: list[tuple[int, int]]        # (top-1 id, top-2 id) at each step
    gaps: list[float]                  # top-1 logit minus top-2 logit at each step


def compare_logits(ours_logits: torch.Tensor, hf_logits: torch.Tensor) -> LogitsResult:
    a, b = ours_logits.float(), hf_logits.float()  # [1, T, vocab]; compare in FP32
    diff = (a - b).abs()
    return LogitsResult(
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        argmax_agree=(a.argmax(-1) == b.argmax(-1)).float().mean().item(),
        top5_same=torch.equal(a[0, -1].topk(5).indices, b[0, -1].topk(5).indices),
    )


@torch.no_grad()
def greedy_no_cache(logits_fn, prompt_ids: torch.Tensor, max_new: int, eos_ids: set[int]) -> GreedyResult:
    """Greedy decoding without a KV cache: every step re-runs the full sequence [1, T + step]."""
    ids = prompt_ids
    out = GreedyResult(tokens=[], top2=[], gaps=[])
    for _ in range(max_new):
        last = logits_fn(ids)[0, -1].float()  # [vocab]: scores for the next token
        vals, idx = last.topk(2)
        nxt = idx[0].item()
        out.tokens.append(nxt)
        out.top2.append((idx[0].item(), idx[1].item()))
        out.gaps.append((vals[0] - vals[1]).item())
        if nxt in eos_ids:
            break
        ids = torch.cat([ids, idx[:1].view(1, 1)], dim=1)
    return out


def first_diff(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, or None if the sequences are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_dtype(name: str, dtype: torch.dtype, args, tok) -> tuple[list[LogitsResult], int, int, int]:
    """Returns (logits results, #greedy matches (main), #greedy matches (hf.generate), #prompts)."""
    t0 = time.perf_counter()
    ours = load_model(args.ckpt, device=args.device, dtype=dtype)
    # transformers 5.x: `dtype=` (not the old `torch_dtype=`). from_pretrained keeps RoPE inv_freq in FP32.
    hf = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=dtype, attn_implementation="sdpa")
    hf = hf.to(args.device).eval()
    print(f"\n=== {name}: models loaded in {time.perf_counter() - t0:.1f}s ===")

    # The checkpoint's generation_config.json says do_sample=True (temperature 0.6, top_k 20, top_p 0.95).
    # Build a fresh greedy config instead of inheriting those sampling settings.
    hf_gen_defaults = GenerationConfig.from_pretrained(args.ckpt)
    eos = hf_gen_defaults.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])  # {151645 <|im_end|>, 151643 <|endoftext|>}
    greedy_cfg = GenerationConfig(
        do_sample=False, max_new_tokens=args.max_new_tokens,
        eos_token_id=sorted(eos_ids), pad_token_id=hf_gen_defaults.pad_token_id,
    )

    ours_fn = lambda ids: ours(ids)                                   # noqa: E731
    hf_fn = lambda ids: hf(input_ids=ids, use_cache=False).logits     # noqa: E731

    logits_results: list[LogitsResult] = []
    n_match = n_match_gen = 0
    for p_i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)  # [1, T], no padding
        with torch.no_grad():
            lr = compare_logits(ours_fn(ids), hf_fn(ids))
        logits_results.append(lr)

        g_ours = greedy_no_cache(ours_fn, ids, args.max_new_tokens, eos_ids)
        g_hf = greedy_no_cache(hf_fn, ids, args.max_new_tokens, eos_ids)
        with torch.no_grad():
            gen = hf.generate(ids, attention_mask=torch.ones_like(ids), generation_config=greedy_cfg)
        gen_tokens = gen[0, ids.shape[1]:].tolist()

        d_main = first_diff(g_ours.tokens, g_hf.tokens)
        d_gen = first_diff(g_ours.tokens, gen_tokens)
        n_match += d_main is None
        n_match_gen += d_gen is None

        print(f"\n[prompt {p_i}] {prompt!r}  ({ids.shape[1]} tokens)")
        print(f"  logits: max abs {lr.max_abs:.3e}, mean abs {lr.mean_abs:.3e}, "
              f"argmax agree {lr.argmax_agree:.1%}, last-pos top-5 same: {lr.top5_same}")
        print(f"  greedy ({len(g_ours.tokens)} tokens) vs HF no-cache: "
              f"{'MATCH' if d_main is None else f'DIFF at step {d_main}'}; "
              f"vs hf.generate: {'MATCH' if d_gen is None else f'DIFF at step {d_gen}'}")
        print(f"  smallest top-1/top-2 gap along our path: {min(g_ours.gaps):.3f}")
        print(f"  ours: {tok.decode(g_ours.tokens)!r}")
        if d_main is not None:
            report_split(tok, d_main, g_ours, g_hf)

    del ours, hf
    gc.collect()
    return logits_results, n_match, n_match_gen, len(PROMPTS)


def report_split(tok, i: int, ours: GreedyResult, hf: GreedyResult) -> None:
    """Explain the first divergence. The prefix before step i is identical, so the gaps are comparable."""
    print(f"  hf:   {tok.decode(hf.tokens)!r}")
    for who, g in (("ours", ours), ("hf  ", hf)):
        if i < len(g.tokens):
            t1, t2 = g.top2[i]
            print(f"  step {i} {who}: top1 {tok.decode([t1])!r} ({t1}), top2 {tok.decode([t2])!r} ({t2}), "
                  f"gap {g.gaps[i]:.4f}")
        else:
            print(f"  step {i} {who}: already stopped (EOS)")
    print("  -> gap near 0 on both sides = rounding near-tie; a large gap = real numerical bug")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=[*DTYPES, "both"], default="both")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    names = list(DTYPES) if args.dtype == "both" else [args.dtype]

    summary = {}
    for name in names:
        summary[name] = run_dtype(name, DTYPES[name], args, tok)

    # Rows ready to paste into the milestone's results table.
    print("\n=== markdown rows ===")
    all_ok = True
    for name, (lrs, n_match, n_gen, n) in summary.items():
        label = name.upper()
        print(f"| 端到端 logits 最大误差（{label}，{n} 条 prompt 取最大） | {max(r.max_abs for r in lrs):.2e} |")
        print(f"| 端到端 logits 平均误差（{label}） | {max(r.mean_abs for r in lrs):.2e} |")
        print(f"| 各位置 argmax 一致率（{label}，最低） | {min(r.argmax_agree for r in lrs):.1%} |")
        print(f"| 贪心 {args.max_new_tokens} token 和 HF 逐 token 一致（{label}，无 cache） | {n_match}/{n} |")
        print(f"| 和 hf.generate（带 KV cache）一致（{label}） | {n_gen}/{n} |")
        all_ok &= n_match == n

    print("\nRESULT:", "PASS" if all_ok else "FAIL (greedy output differs from HF, see above)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
