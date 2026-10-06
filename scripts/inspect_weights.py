"""Print every parameter name / shape / dtype in a safetensors checkpoint.

Usage:
    uv run python scripts/inspect_weights.py checkpoints/Qwen3-0.6B
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def main(ckpt_dir: str) -> None:
    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not files:
        sys.exit(f"no *.safetensors under {ckpt_dir}")

    # name -> (shape, dtype). safe_open only reads the file header, not the tensors.
    params: dict[str, tuple[list[int], str]] = {}
    for path in files:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                params[name] = (sl.get_shape(), sl.get_dtype())

    # Group "model.layers.{i}.xxx" by layer index; everything else is "global".
    layer_re = re.compile(r"model\.layers\.(\d+)\.(.+)")
    layers: dict[int, dict[str, tuple]] = defaultdict(dict)
    global_params = {}
    for name, info in params.items():
        m = layer_re.fullmatch(name)
        if m:
            layers[int(m.group(1))][m.group(2)] = info
        else:
            global_params[name] = info

    print(f"{len(files)} file(s), {len(params)} tensors, {len(layers)} layers\n")

    print("== global ==")
    for name, (shape, dtype) in sorted(global_params.items()):
        print(f"  {name:45s} {dtype:5s} {shape}")

    print("\n== layer 0 ==")
    for name, (shape, dtype) in sorted(layers[0].items()):
        print(f"  {name:45s} {dtype:5s} {shape}")

    # Every layer should look exactly like layer 0.
    bad = [i for i in layers if layers[i] != layers[0]]
    print(f"\nall layers same structure as layer 0: {not bad}" + (f"  (differ: {bad})" if bad else ""))

    total = sum(_numel(shape) for shape, _ in params.values())
    print(f"total parameters: {total / 1e6:.1f}M")
    if "lm_head.weight" in params:
        print("note: checkpoint HAS lm_head.weight (check config.tie_word_embeddings)")


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/Qwen3-0.6B")
