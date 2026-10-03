"""
env_check.py — verify the PyTorch + Triton stack on a Modal GPU and take a
first bandwidth / FLOPs measurement.

Usage:
    modal run scripts/env_check.py              # default: L4
    GPU=A10 modal run scripts/env_check.py      # any Modal GPU string: T4, L4, A10, A100, H100 ...

The first run builds the container image (installs torch), which takes a few
minutes; later runs reuse the cached image.
"""
import os

import modal

GPU = os.environ.get("GPU", "L4")

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")
app = modal.App("mini-vllm-env-check", image=image)

# torch / Triton only exist inside the container (no CUDA build on macOS),
# so the kernel is defined only when the import succeeds.
try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


# Theoretical DRAM bandwidth (GB/s), matched by substring of the device name.
# Order matters: "A100" before "A10", "L40S" before "L4".
PEAK_BW_GBPS = [
    ("H100", 3350),
    ("A100-SXM4-80GB", 2039),
    ("A100", 1555),
    ("A10", 600),  # Modal's A10 is an A10G (24 GB)
    ("L40S", 864),
    ("L4", 300),
    ("T4", 320),
]


@app.function(gpu=GPU, timeout=600)
def check() -> dict:
    import platform
    import subprocess

    assert torch.cuda.is_available(), "CUDA not available in the container"
    props = torch.cuda.get_device_properties(0)

    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
    except FileNotFoundError:
        driver = "n/a"

    info = {
        "GPU": props.name,
        "VRAM (GiB)": round(props.total_memory / 1024**3, 1),
        "SM count": props.multi_processor_count,
        "compute capability": f"{props.major}.{props.minor}",
        "driver": driver,
        "CUDA (torch build)": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }

    # 1) Triton vector add: correctness against torch
    n = 1 << 26  # 64M fp32 elements = 256 MiB per tensor
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    torch.testing.assert_close(out, x + y)
    info["Triton add correct"] = True

    # 2) Bandwidth: read x, read y, write out = 3 * n * 4 bytes
    bytes_moved = 3 * n * x.element_size()
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    ms_triton = triton.testing.do_bench(lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK))
    ms_torch = triton.testing.do_bench(lambda: torch.add(x, y, out=out))
    peak = next((bw for key, bw in PEAK_BW_GBPS if key in props.name), None)
    info["peak DRAM BW (GB/s)"] = peak if peak else "unknown"
    for label, ms in (("Triton add", ms_triton), ("torch add", ms_torch)):
        bw = gbps(ms)
        pct = f" ({bw / peak:.0%} of peak)" if peak else ""
        info[f"{label} BW (GB/s)"] = f"{bw:.0f}{pct}"

    # 3) Compute roof: fp16 matmul TFLOPS (tensor cores)
    m = 4096
    a = torch.randn(m, m, device="cuda", dtype=torch.float16)
    b = torch.randn(m, m, device="cuda", dtype=torch.float16)
    ms_mm = triton.testing.do_bench(lambda: a @ b)
    info["fp16 matmul 4096^3 (TFLOPS)"] = f"{2 * m**3 / (ms_mm * 1e-3) / 1e12:.1f}"

    return info


@app.local_entrypoint()
def main():
    info = check.remote()
    print(f"\nRequested Modal GPU: {GPU}\n")
    print("| metric | value |")
    print("| --- | --- |")
    for k, v in info.items():
        print(f"| {k} | {v} |")
