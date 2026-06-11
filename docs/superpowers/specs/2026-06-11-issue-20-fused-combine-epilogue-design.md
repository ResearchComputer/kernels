# Issue #20 — Fuse the weighted top-k combine into the INT4 W4A16 MoE GEMM epilogue

**Date:** 2026-06-11
**Status:** Approved
**Issue:** ResearchComputer/kernels#20
**Target hardware:** AMD Instinct MI300A (gfx942), HIP-graph-captured decode.

## Purpose

On the decode path the INT4 W4A16 MoE is a chain: `moe_align_block_size` →
`fused_moe_int4_w4a16` (the grouped GEMM, weights folded) → `moe_sum_reduce`. In
xkernels the GEMM writes a token-indexed `[M*top_k, hidden]` buffer and the
interface reduces it with `view(M, top_k, N).sum(1)` — materializing and
re-reading the full top-k expert-output tensor every MoE layer.

This change **fuses the weighted top-k combine into the GEMM epilogue**: each
program accumulates its routing-weighted result directly into the
`[M, hidden]` output via cross-expert atomic add, so the separate
`moe_sum_reduce` launch and the `[M*top_k, hidden]` intermediate disappear. On
Kimi-K2.6 decode (60 MoE layers × every token, `top_k=8`, `hidden=7168`, bf16)
that removes ~`top_k × hidden × 2 B` ≈ 114 KB/token of read+write traffic per
layer — the figure of merit at decode batch (1–16), where these GEMMs are
memory-bound and the launch is free under HIP-graph replay.

`moe_sum_reduce` stays for callers that need the un-reduced per-expert outputs.

## Design

Add an opt-in **`COMBINE`** path to the existing kernel (no body duplication):

- New `COMBINE: tl.constexpr = False` on `_fused_moe_int4_kernel`. When `False`,
  the epilogue is unchanged (`tl.store` to `c[offs_token]`); when `True` it
  computes the output row `offs_token // top_k` and `tl.atomic_add`s the
  weighted accumulator into `c[token, n]`.
- In `COMBINE` mode `c` **is** the `[M, hidden]` **fp32**, zero-initialized
  combine buffer (CDNA3 has native fp32 global `atomic_add`; bf16 atomics are not
  reliable). `compute_type=tl.float32` makes the existing
  `accumulator.to(compute_type)` cast match the fp32 buffer, so `atomic_add`
  operand and pointer dtypes agree. The caller casts the buffer to `A.dtype`
  after the launch.
- Cross-expert correctness: a token routes to `top_k` *distinct* experts, so
  within one expert's block each token appears at most once (no intra-block
  collision), but the same token's rows live in different experts' blocks →
  different program instances write the same `c[token, n_tile]`, which is why the
  atomic is required.
- `MUL_ROUTED_WEIGHT=True` in combine mode (the down combine is weighted); the
  weight is already folded before the epilogue, so the atomic adds the final
  contribution. The fp32 accumulation across experts is *more* accurate than the
  current bf16 `view().sum(1)`.
- The `FILTER_EXPERT` zero-store branch becomes a no-op early-return in `COMBINE`
  mode (the buffer is pre-zeroed); xkernels' wrapper passes `filter_expert=False`
  so this is only for completeness.

### Why one kernel, not two

The combine path differs only in the ~5-line epilogue (address = `offs_token //
top_k`, `atomic_add` vs `store`). A single `COMBINE` constexpr keeps the unpack /
group-scale / `tl.dot` body DRY and lets the #16 tuned configs apply unchanged
(`COMBINE` is not part of the autotune key). Both launch paths (autotuned
fallback and the #16 tuned-direct launch) pass `COMBINE` (default `False`).

## Components

1. **`src/xkernels/ops/moe/triton/moe_int4_kernel.py`**
   - `_fused_moe_int4_kernel`: add `COMBINE: tl.constexpr = False`; branch the
     epilogue (and the filtered-block branch) on it.
   - `int4_w4a16_moe_gemm(..., combine: bool = False)`: thread `COMBINE` through
     both the tuned-direct launch and the autotuned-fallback launch. When
     `combine`, `c` is the `[M, N]` fp32 buffer and `compute_type` is forced to
     `tl.float32`.
   - `_moe_int4_w4a16_triton(..., fused_combine: bool = False)`: when `True`,
     allocate `out = zeros([M, N], float32)`, launch with `combine=True`,
     `mul_routed_weight=True`, and return `out.to(A.dtype)` — no `view().sum(1)`,
     no `[M*top_k, N]` scratch.
2. **`src/xkernels/ops/moe/interface.py`** — `fused_moe_int4_w4a16(...,
   fused_combine: bool = False)` threads the flag to the backend.
3. **`tests/test_moe_int4_w4a16.py`** — add a fused-combine correctness case
   (interpreter + GPU) vs the existing grouped-MoE oracle, both `mul_routed`
   values; assert the `[M, N]` output matches within tolerance.
4. **`benchmarks/bench_moe_combine.py`** + **`slurm/bench_moe_combine_beverin.sbatch`**
   — on-device: time the fused path vs (GEMM into `[M*top_k,N]` + `moe_sum_reduce`)
   at decode M ∈ {1,2,4,8,16} for the down shape (N=7168, K=2048), reporting the
   kernel-count drop (2→1) and the latency/traffic saved.

## Data flow

`COMBINE=False` (default): `acc → store c[offs_token]` → caller `view().sum(1)`
(unchanged). `COMBINE=True`: `acc (weighted, fp32) → atomic_add c[offs_token //
top_k]` into a pre-zeroed `[M,N]` fp32 buffer → caller `.to(bf16)`. The
`moe_sum_reduce` op and the `[M*top_k,N]` tensor are not allocated on the fused
path.

## Error handling / edge cases

- `tl.atomic_add` under `TRITON_INTERPRET=1`: verified early; if the CPU
  interpreter does not support it, the fused-combine correctness test is gated to
  GPU-only (the interpreter still covers the unchanged `COMBINE=False` path).
- Combine buffer must be fp32 and zeroed; the wrapper owns this so callers cannot
  pass a wrong-dtype/dirty buffer.
- `combine=True` with `mul_routed_weight=False` is permitted (un-weighted
  cross-expert sum) but the default + intended use is the weighted down combine.
- Padding slots (`offs_token == pad_id`) are masked out by `token_mask`, so they
  never atomic-add into a valid row.

## Testing

- **CPU / `TRITON_INTERPRET=1`:** fused-combine output equals the oracle (if
  `atomic_add` is supported there); the existing `COMBINE=False` tests prove no
  regression.
- **On device (beverin, gfx942):** fused-combine matches the oracle in bf16
  (atol/rtol 2e-2); the bench reports fused vs GEMM+`moe_sum_reduce` latency and
  the 2→1 kernel-count reduction across decode M.

## Out of scope

- Fusing gate/up + SwiGLU + down into one kernel (xkernels models the MoE as a
  single grouped GEMM; this issue is only the combine epilogue).
- Removing or changing `moe_sum_reduce` (kept for un-reduced consumers).
- Making `COMBINE=True` the default (opt-in only; default path is byte-for-byte
  unchanged).

## Deliverable acceptance

- `fused_moe_int4_w4a16(..., fused_combine=True)` returns `[M, N]` matching the
  GEMM+`moe_sum_reduce` reference within bf16 tolerance, with no `[M*top_k, N]`
  intermediate and no separate reduce kernel.
- Existing INT4 MoE tests unchanged (default `fused_combine=False`).
- On-device bench shows the fused path is no slower (and saves a kernel +
  traffic) at decode M; result reported on #20.
