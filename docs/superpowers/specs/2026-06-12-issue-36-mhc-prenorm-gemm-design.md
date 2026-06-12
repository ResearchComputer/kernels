# Issue #36 — DeepSeek-V4 MHC hidden-compression prenorm GEMM on gfx942

**Date:** 2026-06-12
**Status:** Approved
**Issue:** ResearchComputer/xkernels#36
**Target hardware:** AMD Instinct MI300A (gfx942), ROCm 7.2 / torch 2.11.
**Umbrella:** DeepSeek-V4 MI300A bring-up (#28). Prior gating blocker done in
#32/#33 (sparse-MLA attention compute).

## Purpose

The next gating kernel for serving DeepSeek-V4-Flash on MI300A after sparse-MLA
(#32/#33). With sparse-MLA bound, the V4 forward now reaches the **MHC
(multi-head hidden-compression) layer** and dies there:

```
RuntimeError: deep_gemm.tf32_hc_prenorm_gemm is unavailable
  tokenspeed/runtime/layers/deepseek_v4_mhc.py:233  (mhc_pre)
```

`deep_gemm.tf32_hc_prenorm_gemm` is NVIDIA-only; on AMD it raises. It needs a
portable (Triton / torch) replacement, exactly like sparse-MLA got in #33. This
op is the **GEMM + RMS-prenorm-squared-sum** half of `mhc_pre`; the downstream
TileLang post-fusion (sinkhorn + sigmoid mixing + RMS combine) is already
portable on AMD and is untouched.

## Confirmed facts (from the tokenspeed checkout)

Source: `python/tokenspeed/runtime/layers/deepseek_v4_mhc.py` (the
`amd/deepseek-v4-indexer` branch). The op is called once, in-place:

```python
deep_gemm.tf32_hc_prenorm_gemm(
    residual_flat.view(num_tokens, hc_hidden_size),  # A: [T, K] bf16
    fn,                                              # [N, K] fp32 weights (Linear orientation)
    gemm_out_mul,     # OUT [n_splits, T, N] fp32
    gemm_out_sqrsum,  # OUT [n_splits, T]    fp32
    n_splits,         # int
)
```

with `hc_mult = residual.shape[-2]`, `hidden_size = residual.shape[-1]`,
`K = hc_hidden_size = hc_mult * hidden_size`, and
`N = hc_mult3 = 2*hc_mult + hc_mult**2`. **`fn` is `[N, K]`** (Linear-weight
orientation), pinned from `self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc,
hc_dim))` in `deepseek_v4.py:4143` (`mix_hc = (2+hc_mult)*hc_mult = N`,
`hc_dim = hc_mult*hidden = K`); the op computes **`F.linear(A, fn) = A @ fnᵀ`**,
not `A @ fn`. The torch oracle `hc_head` (`deepseek_v4.py:423`) confirms it:
`mixes = F.linear(x, hc_fn.float())`. For V4-Flash: `hc_mult=4` → `N=24`,
`hidden=4096` → `K=16384`; V4-Pro: `hidden=7168` → `K=28672`. `n_splits` comes
from `_compute_num_split(block_k=64, K, ceil_div(T, block_m=64))` (an occupancy
heuristic; CU-count / grid-size based). The outputs are allocated with
`torch.empty` (uninitialized) → **every split slot must be written**.

**What the GEMM must produce, reverse-engineered from the TileLang consumer**
(`_mhc_pre_big_fuse_tilelang`, lines 83–90):

```python
rms[0] += gemm_out_sqrsum[i_split, i]          # summed over splits, then
rms[0]  = rsqrt(rms[0] / (hc_mult*hidden) + eps)   #   used as the RMS prenorm scale
mixes[j] = Σ_i_split gemm_out_mul[i_split, i, j]   # summed over splits → full A@fn
```

So, letting `Af = A.float()`:
- `Σ_split gemm_out_mul[:, t, :]  ==  F.linear(Af, fn.float())[t, :]`   (full `[T,N]`, = `Af @ fn.float().T`)
- `Σ_split gemm_out_sqrsum[:, t]  ==  (Af * Af).sum(-1)[t]`             (full row sqsum)

**The split partition is numerically irrelevant.** TileLang only ever sums
across the split axis, so any complete, disjoint K-partition into `n_splits`
groups yields identical downstream `mixes`/`rms`. Split-K is purely an occupancy
device. This frees the reference to use the trivial partition and the Triton
backend to use a genuine contiguous-K split.

**Audit (scope item 2 of the issue) — the only NVIDIA-only dep is this op.**
In `deepseek_v4_mhc.py`, `deep_gemm` is imported once (line 15) and called once
(line 284). `mhc_post` and the `mhc_pre` post-fusion are pure TileLang
(`_mhc_post_tilelang`, `_mhc_pre_big_fuse_tilelang`), and the issue confirms
TileLang is available in the AMD image. Replacing this single GEMM unblocks the
whole MHC layer; no other `deep_gemm.*` audit work remains.

## Scope decisions (locked)

- **Everything in one PR**: torch reference oracle + Triton split-K gfx942 kernel
  + offline tests + on-device beverin validation + bench + docs/spec/plan + draft
  PR. Mirrors how #32/#33 shipped (the repo's established bar).
- **API**: a clean xkernels-native op (`hc_prenorm_gemm`, returns the two tensors)
  + a faithful-named in-place wrapper (`tf32_hc_prenorm_gemm`) matching the
  deep_gemm signature exactly, so tokenspeed binds it drop-in (the
  tokenspeed-kernel boundary). The tokenspeed-side binding itself is out of scope
  (a tokenspeed change), as in #33.
- **fp32 compute, not literal TF32**: CDNA3 (gfx942) has no TF32. We compute the
  GEMM and the squared-sum in fp32 (`tl.dot` fp32 accumulate). fp32 is ≥ TF32
  precision; the parity target is our own fp32 reference, not bit-equality with
  NVIDIA deep_gemm. (TF32-vs-fp32 differences are ~1e-3 relative — absorbed if a
  real V4 layer ever cross-checks, cf. #33's 1.95e-3.)
- **General `(K, N, n_splits)`**: support arbitrary `hc_mult`/`hidden` (N and K),
  `n_splits ≥ 1`, K not divisible by `BLOCK_K`, and the `T = 0` edge case.

## Design

New kernel **type** `mhc` under `src/xkernels/ops/` (the op is a GEMM+norm fusion
that fits none of `attention/comm/ffn/moe/norm` cleanly), structured like #33:

```
src/xkernels/ops/mhc/
  reference.py              # oracle + Backend.REFERENCE (registers "hc_prenorm_gemm")
  interface.py              # native op + faithful in-place wrapper
  __init__.py               # re-export + optional triton import (side-effect)
  triton/
    __init__.py
    prenorm_gemm_kernel.py  # Backend.TRITON: split-K GEMM + fused sqsum (gfx942)
```

### Core native op

```python
hc_prenorm_gemm(
    a, fn, *, n_splits, backend="auto",
) -> (gemm_out_mul [n_splits, T, N], gemm_out_sqrsum [n_splits, T])
```

- `a [T, K]` bf16 (the flattened residual; fp32 also accepted), `fn [N, K]` fp32
  weights (Linear orientation), `N = hc_mult3`. `n_splits ≥ 1`.
- Returns the two fp32 split-layout tensors such that `gemm_out_mul.sum(0) ==
  F.linear(a.float(), fn.float())` and `gemm_out_sqrsum.sum(0) ==
  (a.float()**2).sum(-1)`.
- Reference (pure torch, oracle + CPU/no-Triton default): compute the full
  `mul`/`sqrsum` in fp32, write them into split 0, zero splits `1..n_splits-1`
  (a valid complete partition: split 0 covers `[0,K)`, the rest cover empty
  ranges). Numerically the target for the Triton backend.

### Faithful-named wrapper (the tokenspeed binding target)

```python
tf32_hc_prenorm_gemm(a, fn, gemm_out_mul, gemm_out_sqrsum, n_splits) -> None
```

Exact `deep_gemm` signature: writes the two pre-allocated `[n_splits, T, N]` /
`[n_splits, T]` out tensors **in place** and returns `None`. Thin adapter:
dispatch through the native op, copy the results into the provided buffers (or,
for the Triton backend, write them directly). Validates shapes/dtypes against
the buffers.

### Kernel strategy (Triton, gfx942)

Genuine split-K for the tall-skinny GEMM (huge `K`, tiny `N`):

- Grid `(n_splits, ceil_div(T, BLOCK_M))`. Program `(s, m)` owns row-tile `m` and
  the contiguous K-range `[k_lo(s), k_hi(s))` where the `ceil_div(K, BLOCK_K)`
  K-blocks are partitioned as evenly as possible across the `n_splits` programs
  (boundaries cover `[0,K)` disjointly → exact cross-split sum).
- Stream that K-range in `BLOCK_K=64` chunks: load `a_tile [BLOCK_M, BLOCK_K]`
  (bf16→fp32) and a **transposed** `fn_tile [BLOCK_K, N_PAD]` (fp32) — `fn` is
  `[N, K]` so the tile is gathered with K on axis 0, N on axis 1
  (`fn_ptr + k[:,None]*stride_k + n[None,:]*stride_n`), realizing `A @ fnᵀ`.
  Accumulate `acc [BLOCK_M, N_PAD] += tl.dot(a_tile, fn_tile)` **and**
  `sq [BLOCK_M] += sum(a_tile*a_tile, axis=1)` from the same A loads (the fusion
  that motivates the op). `N` padded to a power-of-two `N_PAD ≥ 16` tile with
  column masking; K tail masked (`ks < K`).
- Write `gemm_out_mul[s, m_rows, :N]` and `gemm_out_sqrsum[s, m_rows]`. A split
  whose K-range is empty writes zeros (keeps the `torch.empty` slots defined).
- Correctness-first; one `tl.dot` per K-chunk. (A `BLOCK_M`-only non-split path
  is the `n_splits==1` degenerate case of the same kernel.)

## Testing & validation

- `tests/test_mhc_prenorm_gemm.py`: Triton vs reference oracle, GPU bf16 /
  `TRITON_INTERPRET=1` CPU fp32. Assertions on the **summed** invariant
  (`out_mul.sum(0)` vs `a.float()@fn.float()`, `out_sqrsum.sum(0)` vs
  `(a²).sum(-1)`) — the quantity TileLang consumes. Cases: `n_splits∈{1,4,16}`,
  `hc_mult∈{2,4}` (N∈{8,24}), `hidden` giving K both divisible and not divisible
  by `BLOCK_K`, V4-Flash shape (K=16384, N=24), `T=0` edge, and the faithful
  wrapper's in-place write (buffers mutated, returns None). Tolerances: `1e-4`
  interpreter (fp32), `~2e-2` GPU (bf16 input round), mirroring #33.
- `slurm/test_mhc_prenorm_beverin.sbatch`: on-device gfx942 — `pytest` the file
  with `TRITON_INTERPRET` unset (real compile) + a V4-shape parity max|err|
  standalone check. Mirrors `slurm/test_sparse_mla_beverin.sbatch`.
- `benchmarks/bench_mhc_prenorm_gemm.py` + a README Performance row (speedup vs a
  naive `F.linear(a.float(), fn.float())` + separate `(a²).sum(-1)` torch
  baseline) and a wire into `benchmarks/bench_all.py`.
- `docs/issue-36-mhc-prenorm-gemm.md` (kernel doc: the math, the split invariant,
  the layout, the on-device numbers).

## Public surface

Re-export from `src/xkernels/__init__.py`: `hc_prenorm_gemm`,
`tf32_hc_prenorm_gemm`.

## Out of scope

- tokenspeed-side binding (wiring the AMD path to call the xkernels op instead of
  raising) — a tokenspeed change, not this repo.
- The TileLang post-fusion of `mhc_pre` (`_mhc_pre_big_fuse_tilelang`: sinkhorn,
  sigmoid pre/post mixing, RMS combine) and `mhc_post` — already portable on AMD;
  untouched. Confirmed by the audit: no other `deep_gemm.*` in the MHC path.
- Literal TF32 emulation / bit-equality with NVIDIA `deep_gemm`. We match our own
  fp32 reference.
- Split-K *auto-tuning* of `n_splits` (tokenspeed passes it in via
  `_compute_num_split`); we honor whatever `n_splits` is given.
</content>
</invoke>
