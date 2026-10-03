# mini-vllm

A small LLM inference engine written from scratch — paged KV cache, continuous batching, and hand-written Triton kernels — benchmarked against Hugging Face and vLLM on a single 24 GB GPU.

> **Status:** 🚧 work in progress (started Oct 2026). The roadmap below is updated as milestones land.

## Why

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) showed that a ~1,200-line engine can match vLLM's throughput. This project rebuilds the same core from scratch and goes further in four places:

- **Own Triton paged-attention kernel** instead of relying on FlashAttention / FlashInfer.
- **Ablations for every optimization** (CUDA Graphs, `torch.compile`, paged KV, batching policy): each one measured on and off.
- **Multi-LoRA serving**: serve several LoRA adapters on one base model without merging weights (in the spirit of Punica / S-LoRA).
- **Reproducible benchmarks**: same hardware, same workload, compared against HF `generate`, nano-vllm, and vLLM.

## Models & hardware

- **Models:** Qwen3-0.6B (primary — same setting as nano-vllm's public benchmark), Qwen3-1.7B. The config loader also handles Qwen2.5.
- **Hardware:** single 24 GB GPU on [Modal](https://modal.com) — L4 for day-to-day development, A10 for the reported benchmarks.

## Roadmap

| # | Milestone | Target |
| --- | --- | --- |
| M1 | Project skeleton; load HF weights into my own model definition | Oct 11 |
| M2 | Forward pass numerically matches HF | Oct 18 |
| M3 | KV cache (prefill + decode) and sampling → **v1** | Oct 25 |
| M4 | Continuous batching and scheduler | Nov 8 |
| M5 | Paged KV cache (block manager + block tables) | Nov 15 |
| M6 | Triton paged-attention kernel | Nov 29 |
| M7 | CUDA Graphs + `torch.compile` for decode; multi-LoRA serving | Dec 13 |
| M8 | Benchmark & profiling report | Dec 20 |
| M10 | OpenAI-compatible API server (`/v1/chat/completions`, streaming) | Jan 3 |

## Repository layout (planned)

```
mini_vllm/      engine: model, KV cache, scheduler, sampler, LLM entrypoint
kernels/        Triton kernels (softmax, LayerNorm, matmul, fused & paged attention) with tests + benchmarks
benchmarks/     benchmark scripts and results
scripts/        utilities (environment check, Modal launchers)
tests/          correctness tests against Hugging Face
```

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`. GPU work runs on Modal, whose container image is built from the same lock file, so the local and GPU environments match.

```bash
uv sync                                          # create .venv with Python 3.12 + locked deps
uv run modal setup                               # one-time browser login
uv run modal run scripts/env_check.py            # PyTorch + Triton sanity check on an L4
GPU=A10 uv run modal run scripts/env_check.py    # same check on an A10
```

`env_check.py` prints the GPU, driver, and torch/Triton versions, checks a Triton vector-add kernel against PyTorch, and measures achieved DRAM bandwidth and fp16 matmul TFLOPS.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022)
- Dao et al., *FlashAttention* (NeurIPS 2022)
- [vLLM](https://github.com/vllm-project/vllm), [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
