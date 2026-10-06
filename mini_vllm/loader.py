"""Load a Hugging Face safetensors checkpoint into our own Qwen3 model (no HF model classes)."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from mini_vllm.config import Qwen3Config
from mini_vllm.models.qwen3 import Qwen3ForCausalLM


def load_model(
    ckpt_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Qwen3ForCausalLM:
    """Build the model directly on `device` in `dtype` and fill it with checkpoint weights."""
    config = Qwen3Config.from_pretrained(ckpt_dir)
    dtype = dtype or getattr(torch, config.torch_dtype)  # "bfloat16" -> torch.bfloat16

    # 1) Build on "meta": shapes only, no memory, no random init (which would be thrown away anyway).
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    # 2) Allocate real (uninitialized) storage on the target device, already in the target dtype.
    model = model.to(dtype).to_empty(device=device)
    model.tie_weights()  # to_empty() creates new tensors, so re-tie lm_head to the embedding
    # 3) Overwrite every parameter with the checkpoint values.
    load_weights(model, ckpt_dir)
    return model.eval()


@torch.no_grad()  # copying weights is not a training step: don't track gradients
def load_weights(model: Qwen3ForCausalLM, ckpt_dir: str | Path) -> None:
    """Copy every tensor from `ckpt_dir/*.safetensors` into the parameter with the same name.

    Fails loudly if any parameter is missing, any checkpoint tensor is unexpected,
    or any shape differs -- a silently half-loaded model produces garbage that is hard to debug.
    """
    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {ckpt_dir}")

    # named_parameters() de-duplicates shared tensors: when tied, "lm_head.weight" is NOT in here
    # (it is the same Parameter as "model.embed_tokens.weight").
    params = dict(model.named_parameters())
    tied = model.config.tie_word_embeddings

    loaded: set[str] = set()
    unexpected: list[str] = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in params:
                    # Qwen3-0.6B ships a redundant lm_head.weight even though it is tied; skip it.
                    if not (tied and name == "lm_head.weight"):
                        unexpected.append(name)
                    continue
                tensor = f.get_tensor(name)
                param = params[name]
                if tensor.shape != param.shape:
                    raise ValueError(f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}")
                param.copy_(tensor)  # copy_ also converts dtype / moves to the param's device
                loaded.add(name)

    missing = sorted(params.keys() - loaded)
    if missing or unexpected:
        raise ValueError(f"checkpoint does not match model: missing={missing[:5]}... ({len(missing)}), "
                         f"unexpected={unexpected[:5]}... ({len(unexpected)})")
