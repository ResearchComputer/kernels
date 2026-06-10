# xkernels

Customized compute kernels across hardware vendors (NVIDIA, AMD, …) and kernel
types (FFN, MoE, comm, …), with a uniform PyTorch API, automatic backend
dispatch, and a correctness + benchmark harness.

## Install

```bash
pip install -e ".[dev]"          # pure-Python (reference + triton if present)
XKERNELS_FORCE_BUILD=1 pip install -e .   # also build CUDA/HIP extensions
```

Triton/CUDA backends are optional; the package runs on the pure-torch reference
path anywhere.

## Usage

```python
import torch
from xkernels import fused_ffn

y = fused_ffn(x, w_gate, w_up, w_down)            # backend="auto"
y = fused_ffn(x, w_gate, w_up, w_down, backend="triton")  # force a backend
```

```python
from xkernels import fused_moe_int4_w4a16  # INT4 W4A16 grouped fused-MoE GEMM

out = fused_moe_int4_w4a16(A, packed, scale, topk_ids, topk_w, group_size=32)
```

Override globally with `XKERNELS_BACKEND=reference|triton|cuda|hip`.

## Performance

Speedup of each kernel's optimized backend over the naive PyTorch a practitioner
would write without it, on one **AMD Instinct MI300A (gfx942)**, bf16, median of
Triton `do_bench`. Reproduce with `python benchmarks/bench_all.py` (single GPU)
or `sbatch slurm/bench_all_beverin.sbatch`.

| Kernel | Shape | Naive PyTorch | Optimized | Speedup |
|--------|-------|--------------:|----------:|--------:|
| `moe_int4_w4a16` | M=64, E=48, N=4096, K=7168, top_k=8 | 32.8 ms (dequant+matmul) | 2.02 ms | **16.3×** |
| `moe_sum_reduce` | M=8192, top_k=8, H=7168 | 3.21 ms (torch reduce) | 0.38 ms | **8.4×** |
| `dual_rmsnorm` | T=8192, d=(1536,512) | 0.25 ms (2× sequential RMSNorm) | 0.05 ms | **5.0×** |
| `mha_merge_state` | T=8192, H=128, D=128 | 2.57 ms (torch merge) | 0.80 ms | **3.2×** |
| `fused_ffn` | M=4096, 4096→11008 (fp16) | 5.56 ms (unfused torch) | 5.49 ms | **1.0×** |

Naive baselines: `moe_int4_w4a16` vs per-expert dequant(int4→bf16)+matmul;
`moe_sum_reduce` / `mha_merge_state` vs their torch oracles; `dual_rmsnorm` vs
two sequential RMSNorm launches; `fused_ffn` vs the unfused `reference` backend.

Notes:

- **`fused_ffn` ≈ 1.0×** — the Triton backend fuses only the SwiGLU *activation*;
  the three projection GEMMs dominate and are torch matmuls in both paths, so
  there is little left to win. Measured in fp16 because on this torch
  2.11+rocm7.2 build the **bf16** GEMM misses the MFMA/hipBLASLt path and runs
  ~470× slower than fp16 (0.8 vs 358 TFLOP/s; see `benchmarks/probe_ffn.py`) —
  a stack issue, not a kernel one.
- **`moe_align_block_size`** ships a reference backend only; the Triton perf
  kernel (device-atomic histogram + padded prefix-sum) is a tracked follow-up
  (issue #4), so there is no speedup to report yet.
- **`hierarchical_all_reduce`** (distributed) does *not* beat a flat all-reduce on
  the 2-node / 4-NIC-per-node MI300A stack — RCCL's flat collective is already
  topology-aware. Full analysis in `docs/issue-12-hierarchical-all-reduce.md`.

## Layout

- `src/xkernels/ops/<type>/` — kernels by type; each has `reference.py`,
  `interface.py`, and per-backend subdirs (`triton/`, `cuda/`).
- `src/xkernels/_dispatch.py` — backend registry + selection.
- `tests/`, `benchmarks/`, `examples/` — harness and demos.

See `docs/adding-a-kernel.md` to extend. Design: `docs/superpowers/specs/`.
