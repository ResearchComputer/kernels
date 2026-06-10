# kernels

Standalone, tuned + microbenchmarked GPU kernels extracted from the tokenspeed
serving stack, targeting **AMD MI300A (gfx942, CDNA3)** (portable to NVIDIA where
practical). Each kernel here corresponds to a tracking issue and ships as a
self-contained directory with a slow-but-correct PyTorch baseline, the optimized
kernel, a numerical correctness check against the baseline, and a microbenchmark.

## Layout convention

One directory per kernel:

```
<kernel_name>/
  README.md            compute-pattern doc + optimization notes + how to run
  reference.py         slow but correct PyTorch baseline (the numerical oracle)
  kernel.py            optimized kernel + Python launcher
  configs.py           autotune config space (for autotuned kernels)
  test_correctness.py  numerical check vs reference.py
  benchmark.py         microbenchmark over the relevant production shapes
```

Backend choice follows tokenspeed's `AGENTS.md`: prefer **Triton Gluon** for AMD
GPU kernels, **CuteDSL** for NVIDIA, and **portable Triton** for cross-vendor
solutions. Vendor libraries stay optional.

## Kernels

Tracked by issues #1-#5 (INT4 W4A16 fused-MoE GEMM, dual RMSNorm,
`mha_merge_state`, `moe_align_block_size`, `moe_sum_reduce`). Each lands as a PR
adding its own directory.

## Testing without a GPU

Kernels here run under the Triton CPU interpreter for correctness:

```bash
TRITON_INTERPRET=1 pytest <kernel_name>/test_correctness.py
```

Microbenchmarks require a real GPU and must be run on a node you already hold —
they never submit a cluster job.
