import json
import struct
from pathlib import Path

import pytest
import torch

from mini_vllm.config import Qwen3Config
from mini_vllm.models.qwen3 import Qwen3ForCausalLM
from tests.test_config import QWEN3_0_6B


def build_on_meta(cfg: Qwen3Config) -> Qwen3ForCausalLM:
    # "meta" tensors have shape/dtype but no storage: building the 0.6B model costs ~0 memory.
    with torch.device("meta"):
        return Qwen3ForCausalLM(cfg)


def test_parameter_shapes():
    cfg = Qwen3Config.from_dict(QWEN3_0_6B)
    shapes = {k: list(v.shape) for k, v in build_on_meta(cfg).state_dict().items()}

    assert len(shapes) == 28 * 11 + 3  # same 311 tensors as the checkpoint
    assert shapes["model.embed_tokens.weight"] == [151936, 1024]
    assert shapes["lm_head.weight"] == [151936, 1024]
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == [2048, 1024]
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == [1024, 1024]
    assert shapes["model.layers.0.self_attn.o_proj.weight"] == [1024, 2048]
    assert shapes["model.layers.0.self_attn.q_norm.weight"] == [128]
    assert shapes["model.layers.27.mlp.down_proj.weight"] == [1024, 3072]


def test_tied_embeddings_share_storage():
    model = build_on_meta(Qwen3Config.from_dict(QWEN3_0_6B))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # named_parameters() de-duplicates shared tensors -> unique count is 596M, not 751.6M.
    n = sum(p.numel() for p in model.parameters())
    assert round(n / 1e6) == 596


CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-0.6B"


def read_safetensors_header(path: Path) -> dict:
    """First 8 bytes = header length (little-endian u64), then a JSON header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="checkpoint not downloaded")
def test_matches_real_checkpoint():
    cfg = Qwen3Config.from_pretrained(CKPT)
    ours = {k: list(v.shape) for k, v in build_on_meta(cfg).state_dict().items()}
    theirs = {k: v["shape"] for k, v in read_safetensors_header(CKPT / "model.safetensors").items()}
    assert ours.keys() == theirs.keys(), (ours.keys() - theirs.keys(), theirs.keys() - ours.keys())
    assert ours == theirs
