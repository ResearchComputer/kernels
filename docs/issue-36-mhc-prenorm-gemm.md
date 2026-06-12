# Issue #36 — DeepSeek-V4 MHC hidden-compression prenorm GEMM (gfx942)

This kernel replaces `deep_gemm.tf32_hc_prenorm_gemm` — the GEMM-plus-prenorm-accumulation
half of DeepSeek-V4's `mhc_pre` operator — with a portable, AMD-native Triton
implementation. The blocking dependency before this work was the sparse-MLA attention
compute (#32/#33), now resolved; issue #36 is the next gating kernel on the path to
full MHC support on gfx942 (umbrella #28).

The new xkernels-native op is `hc_prenorm_gemm`. For drop-in compatibility it is
re-exported under the upstream-faithful name `tf32_hc_prenorm_gemm`.

## The math

Given a token batch reshaped to `A [T, K]` (bf16, from `residual.view(T, K)`) and the
combined hidden-compression weight matrix `fn [N, K]` (fp32, Linear orientation), the
operation computes two quantities simultaneously:

```
gemm_out    = F.linear(A, fn)   # A @ fn.T,  shape [T, N]
sqrsum_out  = Σ_k A[t, k]²      # per-row sum-of-squares,  shape [T]
```

Both are computed in fp32 (V4-Flash CDNA3 has no TF32 acceleration; the kernel uses
plain fp32 dot products throughout).

**V4-Flash dimensions.** With `hc_mult=4` and `hidden=4096`:

| Symbol   | Value | Meaning                          |
|----------|-------|----------------------------------|
| `K`      | 16384 | `hc_mult × hidden`               |
| `N`      | 24    | `2*hc_mult + hc_mult² = 2·4+16`  |
| dtype(A) | bf16  | residual input                   |
| dtype(fn)| fp32  | weight matrix                    |

This is a memory-bound tall-skinny GEMM (huge K=16384, tiny N=24). The dominant cost
is streaming the K dimension; fusing `Σ A²` on the same A loads is essentially free.

## Split-K layout and the summed invariant

To exploit occupancy for small decode batch sizes (T ≪ 1024), the K dimension is
partitioned into `n_splits` contiguous blocks. Each Triton program handles one
`(split_idx, row_tile)` pair, operating on a contiguous K-range of length `⌈K/n_splits⌉`
(the last split is zero-padded when K is not divisible).

The two output tensors carry a **split axis in position 0**:

- `gemm_out_mul  [n_splits, T, N]` — partial GEMM accumulations
- `gemm_out_sqrsum [n_splits, T]` — partial squared-sum accumulations

**Key invariant:** the downstream TileLang post-fusion step only ever sums across the
split axis. Summing recovers the full results:

```python
gemm_out_mul.sum(0)    == F.linear(A.float(), fn)   # shape [T, N]
gemm_out_sqrsum.sum(0) == (A.float() ** 2).sum(-1)  # shape [T]
```

Because the downstream consumer only sums, the K-partition is numerically free — any
assignment of K indices to splits produces the same final values. Split-K exists purely
for occupancy: parallelising the K=16384 reduction over many CUs improves utilisation
at the small T values typical of autoregressive decode.

Empty splits (when `n_splits > ⌈K/BLOCK_K⌉`) store explicit zeros; the sum invariant
still holds.

## Audit: unblocking the MHC layer

The `deep_gemm` dependency sits at `deepseek_v4_mhc.py:284` (the `mhc_pre` prenorm
GEMM call). It is the **only** NVIDIA-only component in the entire MHC forward path:

- `mhc_post` is pure TileLang — already portable on AMD.
- The `mhc_pre` post-fusion (the normalization and projection steps that consume
  `gemm_out_mul` and `gemm_out_sqrsum`) is also pure TileLang.

Replacing this one call with `hc_prenorm_gemm` removes the last NVIDIA-only barrier
and makes the complete MHC layer runnable on gfx942.

## Kernel strategy

One Triton program per `(split_idx, row_tile)`:

1. **K-range selection.** Each program computes its contiguous K slice:
   `k_start = split_idx * split_k_size`, `k_end = min(k_start + split_k_size, K)`.
   An empty slice (k_start ≥ K) stores zeros and exits.
2. **Fused load and compute.** The inner loop tiles over the K range in `BLOCK_K`
   steps. Each iteration loads a tile of A `[BLOCK_M, BLOCK_K]` (bf16 → fp32) and a
   tile of fn `[BLOCK_N, BLOCK_K]` (fp32). A single `tl.dot(A_tile, fn_tile.T)`
   accumulates into the GEMM result; `(A_tile * A_tile).sum(-1)` accumulates into the
   squared-sum result — both from the **same** A load with no extra memory traffic.
3. **Store.** The partial GEMM result is written to `gemm_out_mul[split_idx, row, :]`
   and the partial squared-sum to `gemm_out_sqrsum[split_idx, row]`.

## Validation

Offline tests live in `tests/test_mhc_prenorm_gemm.py` and cover:

- **Interpreter mode** (`TRITON_INTERPRET=1`, CPU fp32): verifies the sum invariant
  `mul.sum(0) ≈ F.linear(A, fn)` and `sqr.sum(0) ≈ (A²).sum(-1)` without a GPU.
- **GPU mode** (gfx942): same assertions at bf16 A / fp32 fn, multiple T values.
- **K-not-divisible:** `n_splits` chosen so K is not evenly divisible; checks no
  off-by-one in the last split.
- **Empty-split:** `n_splits` deliberately larger than `⌈K/BLOCK_K⌉`; verifies that
  zero-padded splits do not corrupt the sum.
- **T=0:** zero-token batch; checks the kernel exits cleanly and returns empty tensors
  of the right shape.
- **Faithful wrapper:** `tf32_hc_prenorm_gemm` is called with the same inputs and
  compared to `hc_prenorm_gemm`; must match exactly (it is a thin re-export).

On-device run (beverin, AMD Instinct MI300A / gfx942):

| Check | Shape | Result |
|-------|-------|--------|
| `pytest tests/test_mhc_prenorm_gemm.py` | — | *(filled by beverin run — issue #36 Task 6)* |
| sum invariant (bf16→fp32) | T=8, K=16384, N=24, splits=16 | *(filled by beverin run — issue #36 Task 6)* |
| benchmark vs F.linear+sqsum | T=1/8/64, K=16384, N=24 | *(filled by beverin run — issue #36 Task 6)* |

> On-device numbers are placeholders pending the Task 6 beverin run. No performance
> figures are reported here until measured on real hardware.
