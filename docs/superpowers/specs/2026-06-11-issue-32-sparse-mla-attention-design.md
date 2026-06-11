# Issue #32 — DeepSeek-V4 sparse-MLA attention compute on gfx942

**Date:** 2026-06-11
**Status:** Approved
**Issue:** ResearchComputer/xkernels#32
**Target hardware:** AMD Instinct MI300A (gfx942), ROCm 7.2 / torch 2.11.
**Umbrella:** DeepSeek-V4 MI300A bring-up (#28). Selection/indexer side done in
#27/#31 (`dsa_indexer_logits`/`dsa_indexer_topk`) and #29 (mxfp4 paged gather).

## Purpose

The last gating blocker for serving DeepSeek-V4 on MI300A: the V4 **sparse-MLA
attention compute** has no gfx942 implementation. tokenspeed binds
`flash_mla_sparse_fwd` / `flash_mla_with_kvcache` / `get_mla_metadata` to
DeepSeek's NVIDIA-only `flash_mla` package **only on Hopper+**; on AMD they are
`error_fn`, so `forward_deepseek_v4_prefill`/`_decode` raise. The
indexer→selection→**[this kernel]**→o_proj chain is otherwise complete.

This kernel **consumes** the DSA indexer's top-k KV indices and runs the actual
Compressed/Heavily-Compressed/Sliding-Window attention softmax over V4's latent
KV, returning `(out, lse)`.

## Scope decisions (locked)

- **Everything in one PR**: reference oracle + both Triton kernels (prefill
  ragged + decode paged) + `get_mla_metadata` + on-device beverin validation.
- **API**: a clean xkernels-native op, re-exported under the upstream-faithful
  names so tokenspeed binds them drop-in.
- **Variant-agnostic compute**: the kernel attends over whatever top-k indices it
  is handed. The CSA/HCA/SWA distinction is *selection*-side (indexer #27/#31 +
  the per-ratio gather), not the softmax — confirmed by the runtime, which passes
  pre-selected index sets and per-query valid lengths into the compute call.
- **Include fp8_ds_mla dequant-on-load** in the decode path (the production
  cache format), structured as a separable module so it can be severed if needed.

## Confirmed facts (from the tokenspeed checkout)

Call sites: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`.

- **`flash_mla_sparse_fwd`** (prefill, `:1465`) returns a **3-tuple**
  `(out, max_logits, lse)`:
  ```python
  out, _, _ = flash_mla_sparse_fwd(
      q=q_padded,                       # [T, padded_heads, D]
      kv=kv_workspace.view(-1, 1, D),   # [Kv, 1, D]  bf16 latent MQA workspace
      indices=indices.unsqueeze(1),     # [T, 1, topk] int32 into kv
      sm_scale=softmax_scale,           # float
      attn_sink=attn_sink,              # per-head sink logit(s)
      topk_length=lens,                 # [T] int32 valid count per query
  )
  ```
  `kv_workspace` is already bf16-gathered/dequantized by
  `deepseek_v4_dequantize_and_gather_k_cache` — so prefill compute takes bf16 KV.
- **`flash_mla_with_kvcache`** (decode, `:1057`) returns `(out, lse)` and takes
  **two** caches combined in one softmax:
  ```python
  out, _ = flash_mla_with_kvcache(
      q=q_padded.unsqueeze(1),          # [B, 1, padded_heads, D]
      k_cache=swa_cache,                # fp8_ds_mla view [num_blocks, blk, 1, row_bytes]
      block_table=None, cache_seqlens=None,
      head_dim_v=head_dim,
      tile_scheduler_metadata=get_mla_metadata()[0],
      softmax_scale=softmax_scale, is_fp8_kvcache=True,
      indices=swa_indices.unsqueeze(1),       # SWA selected positions
      attn_sink=attn_sink,
      extra_k_cache=compressed_cache,         # compressed CSA cache (or None)
      extra_indices_in_kvcache=extra_indices,
      topk_length=swa_lens, extra_topk_length=extra_lens,
  )
  ```
- **`get_mla_metadata()`** is called **no-arg** in V4; `[0]` is threaded into the
  decode kernel as `tile_scheduler_metadata`.

V4 latent dims (`tokenspeed_kernel/ops/attention/triton/deepseek_v4.py`):
`DEEPSEEK_V4_HEAD_DIM=512` (full latent qk `D`), `ROPE_DIM=64` (decoupled rope,
score-only, stored bf16), `NOPE_DIM=448` (the `kv_lora`/`c_kv`, value-bearing,
stored fp8 e4m3 with per-`FP8_QUANT_BLOCK=64` fp32 scales). So **`D=512`,
value `d_v=448`**; MQA (one shared latent KV head). topk = 512 (Flash) / 1024
(Pro). No torch oracle for the *compute* exists upstream — we author it.

fp8_ds_mla row (per token): `NOPE_DIM` fp8 + `SWA_SCALE_DIM = NOPE_DIM//64 + 1`
fp32 scales + `ROPE_DIM` bf16; `SWA_TOKEN_STRIDE = NOPE_DIM + ROPE_DIM*2`. The
decode dequant-on-load mirrors this exactly (pinned against the cache writer).

## Design

New sparse-MLA compute under `src/xkernels/ops/attention/`:

```
ops/attention/
  sparse_mla_reference.py     # oracle + Backend.REFERENCE (authored math)
  sparse_mla.py               # fp8_ds_mla layout consts + dequant helper (cf. gather/mxfp4.py)
  interface.py  (extend)      # native op + 3 faithful-named wrappers
  triton/sparse_mla_kernel.py # Backend.TRITON: prefill + decode flash kernels
  __init__.py   (extend)      # re-export + optional triton import
```

### Core native op

```python
sparse_mla_attention(
    q, kv, indices, *,
    sm_scale, topk_length=None, attn_sink=None, d_v=None, backend="auto",
) -> (out, lse, max_logits)
```

- `q [T, H, D]` latent queries (D=512). `kv [Kv, D]` shared latent MQA cache
  (bf16). `indices [T, topk]` int32 into `kv`. Validity via `topk_length[t]`
  (count) **and** `-1` sentinel — support both (V4 uses lengths).
- Per `(t,h)`: `s_j = sm_scale * (q[t,h] · kv[idx_j])` over selected `idx_j`;
  online-softmax including an optional `attn_sink[h]` logit (joins the denom,
  contributes no value); `out[t,h] = Σ_j p_j * kv[idx_j, :d_v]` (value = first
  `d_v=448` latent dims). Return `lse` and `max_logits` (fp32).
- Reference is pure torch (oracle + CPU/no-Triton default), gather-then-masked-
  softmax; numerically the target for the Triton backend.

### Faithful-named wrappers (thin adapters over the core op)

- `flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink=None, topk_length=None)
  -> (out, max_logits, lse)` — squeeze the MQA-head axis on `kv`/`indices`, call
  the core op, return the 3-tuple in upstream order. Matches `deepseek_v4.py:1465`.
- `flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
  tile_scheduler_metadata, *, softmax_scale, is_fp8_kvcache=True, indices,
  attn_sink=None, extra_k_cache=None, extra_indices_in_kvcache=None,
  topk_length=None, extra_topk_length=None) -> (out, lse)` — fp8_ds_mla
  dequant-on-load (448 fp8 nope w/ per-64 fp32 scale + 64 bf16 rope), gather, and
  attend over the **union** of `k_cache`(SWA) ∪ `extra_k_cache`(compressed CSA)
  index sets in one softmax. Matches `deepseek_v4.py:1057`. With `extra_k_cache
  is None` it degenerates to single-cache decode.
- `get_mla_metadata(*args, **kwargs) -> (tile_metadata, num_splits)` —
  lightweight; the compute is correct without split-KV scheduling, so returns a
  small placeholder tensor + `num_splits=1`. The no-arg V4 call works. Shaped so a
  future split-KV path can reuse `mha_merge_state` (#3).

### Kernel strategy

**One Triton program per `(token, head)`**, looping over `topk` in `BLOCK_N`
chunks with online (flash) softmax — correct and simple for topk ≤ 1024, D=512,
d_v=448. Running max + denom + `[d_v]` fp32 accumulator per program; gather KV
rows via `indices` (prefill: from the bf16 workspace; decode: resolve through the
cache + dequant fp8 on load). Sink folds in as one extra logit before the final
normalize. The decode dual-cache path iterates both index sets into the same
online-softmax state.

Alternatives considered and deferred (kernel kept shaped for them):
**split-KV + `mha_merge_state` merge** (better occupancy for long topk) and
**GEMM-tiled `tl.dot`** (overkill at decode M). Correctness-first single-program
now; `get_mla_metadata` + `mha_merge_state` make split-KV a clean follow-up.

## Testing & validation

- `tests/test_sparse_mla_attention.py`: Triton vs oracle, GPU bf16 /
  `TRITON_INTERPRET=1` CPU fp32. Cases: padded `topk_length`, `-1` sentinels,
  sink on/off, single- and dual-cache decode, V4 shapes (H, D=512, d_v=448,
  topk∈{512,1024}). fp8_ds_mla dequant spot-check vs hand-computed values
  (mirrors `test_mxfp4_paged_gather.py`). Tolerances: `1e-4` interpreter (fp32),
  `2e-2` GPU (bf16 round), set/Jaccard not needed (dense output, not selection).
- `slurm/test_sparse_mla_beverin.sbatch`: on-device gfx942 — `pytest` the test
  file with `TRITON_INTERPRET` unset (real compile) + a V4-shape parity + max|err|
  standalone check. Mirrors `slurm/test_dsa_indexer_beverin.sbatch`.
- `benchmarks/bench_sparse_mla.py` + a README Performance row (speedup vs a naive
  gather+dense-softmax torch baseline).
- `docs/issue-32-sparse-mla-attention.md` (kernel doc, the math + layout).

## Public surface

Re-export from `src/xkernels/__init__.py`: `sparse_mla_attention`,
`flash_mla_sparse_fwd`, `flash_mla_with_kvcache`, `get_mla_metadata`.

## Out of scope

- tokenspeed-side binding (replacing `error_fn` with the xkernels ops) — a
  tokenspeed change, not this repo.
- Split-KV scheduling / a non-trivial `get_mla_metadata` work-partition.
- The indexer/selection and the fp8/mxfp4 *gather for the indexer* (done: #27/#31,
  #29). This op only adds the attention **compute** (and its decode dequant).
- mHC, MTP, YaRN, routing — model-level, not this kernel.
