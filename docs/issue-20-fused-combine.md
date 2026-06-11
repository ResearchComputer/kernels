# Issue #20 — Fused top-k combine epilogue: measured, kept opt-in (off by default)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2.
**Bench:** `benchmarks/bench_moe_combine.py` (`slurm/bench_moe_combine_beverin.sbatch`),
job 381848. Metric: median ms, Triton `do_bench`, bf16.

## TL;DR

The fused weighted top-k combine (atomic-accumulate each expert's down-proj result
directly into `[M, hidden]`) is **implemented and numerically correct**, but on
MI300A at decode shapes it is **~25–35% slower** than the unfused `GEMM +
moe_sum_reduce` path. It is therefore shipped **opt-in and off by default**
(`fused_combine=False`); the separate `moe_sum_reduce` kernel stays the
recommended path. (Same outcome shape as issue #12's hierarchical all-reduce: a
correct optimization that the hardware does not reward here.)

## Result (down GEMM: E=48, N=7168, K=2048, top_k=8; align built once, outside timing)

| M | GEMM + moe_sum_reduce (ms) | fused combine (ms) | speedup | max\|err\| |
|--:|---------------------------:|-------------------:|--------:|-----------:|
| 1  | 0.1236 | 0.1603 | 0.77× | 0.0065 |
| 2  | 0.2365 | 0.2989 | 0.79× | 0.0082 |
| 4  | 0.3425 | 0.4512 | 0.76× | 0.0105 |
| 8  | 0.5581 | 0.7541 | 0.74× | 0.0107 |
| 16 | 0.6769 | 0.9010 | 0.75× | 0.0099 |

Both paths build the `moe_align_block_size` dispatch **once** outside the timed
region (it is a separate kernel), so this isolates the combine.

## Why fusion loses here

The grouped GEMM tiles the output **by expert**. A token routes to `top_k=8`
distinct experts, so each token's `[1, hidden]` row is produced by 8 different
program instances. Fusing the combine means those 8 instances must
`atomic_add` into the same `[token, n-tile]` addresses:

1. **Atomic contention.** Per N-tile, 8 atomic adds target the same row and
   serialize. The unfused `moe_sum_reduce` kernel instead reads the 8 per-expert
   values and sums them in registers — no atomics.
2. **fp32 write traffic.** CDNA3 has native fp32 (not bf16) global `atomic_add`,
   so the combine buffer is fp32 — 2× the write bytes of the bf16 intermediate.
3. **Pre-zero.** The fp32 buffer must be zeroed before the adds (an extra memset);
   the unfused GEMM just overwrites its `[M*top_k, hidden]` scratch.

The dedicated `moe_sum_reduce` (issue #5, already 8.3× vs the torch reduce) is fast
enough that eliminating it does not pay for the atomic + fp32 costs at decode `M`.
The issue's traffic argument (skip the `[M*top_k, hidden]` round-trip) is real, but
it is outweighed by the atomic-accumulate overhead on this stack.

## What ships

- `fused_moe_int4_w4a16(..., fused_combine=True)` — correct, opt-in, **off by
  default**. The kernel gains a `COMBINE` constexpr; `COMBINE=False` (default) is
  byte-for-byte the prior behavior. `moe_sum_reduce` is unchanged and remains the
  recommended combine.
- Correctness is covered on CPU (`TRITON_INTERPRET=1`) and GPU
  (`test_fused_combine_matches_reference`, both `mul_routed` values).

## When it might still help (not measured here)

- A stack without an efficient `moe_sum_reduce`, or where the combine is a heavy
  *separate* launch (the production tokenspeed framing).
- Much larger `M` where the GEMM dominates and the relative combine cost shrinks.
- A non-atomic fused design (group a token's `top_k` experts into one block and
  reduce in-kernel) — a different kernel structure than the grouped-by-expert
  GEMM, out of scope here.

These would need their own measurement before flipping the default.
