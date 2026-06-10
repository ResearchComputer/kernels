# INT4 W4A16 grouped fused-MoE GEMM (gfx942)

Optimized in-kernel-dequant INT4 (W4A16) grouped fused-MoE GEMM for **AMD MI300A
(gfx942, CDNA3)**. Tracks [issue #1](../../../issues/1). Serves the routed-expert
GEMM of compressed-tensors `pack-quantized` INT4 MoE models (Kimi-K2.6 /
DeepSeek-V3-class) while keeping weights packed (~4x memory saving vs bf16
dequant).

## Compute pattern

Per expert `e`, weight `W_e` is INT4, **symmetric group-quantized along K**
(`group_size = 32`), stored `uint4b8` (unsigned nibble, subtract 8 -> signed
`[-8, 7]`).

| tensor | shape | dtype | layout |
|--------|-------|-------|--------|
| `B` (packed weights) | `[E, N, K // 8]` | int32 | 8 nibbles / int32, **low nibble = lowest K** |
| `S` (group scales)   | `[E, N, K // group]` | bf16 | one scale per (N, K-group) |
| `A` (activations)    | `[M, K]` | bf16 | token rows (pre-permute) |

```
W_e[n, k] = (((B[e, n, k // 8] >> (4 * (k % 8))) & 0xF) - 8) * S[e, n, k // group]
```

Fused MoE: activations routed to top-k experts; per (token, expert) compute
`A @ W_e^T` over the expert's token block (grouped GEMM driven by
`sorted_token_ids` / `expert_ids`), unpacking + scaling **inline**, accumulating
in fp32. Two GEMMs per layer: gate_up (`N = 2*inter`, `K = hidden`) and down
(`N = hidden`, `K = inter`).

Output is **token-indexed**: `c[m*top_k + j] = A[m] @ W_e^T` (`e = topk_ids[m,j]`),
so the downstream reduction is `c.view(M, top_k, N).sum(1)` (routing weight is
folded into the down GEMM via `mul_routed_weight`).

## Files

| file | what |
|------|------|
| `reference.py` | slow but correct PyTorch oracle: unpack -> dequant -> grouped GEMM, plus `moe_align_block_size_ref` dispatch and a weight generator |
| `kernel.py` | optimized autotuned Triton kernel (`fused_moe_int4_kernel`) + launcher (`int4_w4a16_moe_gemm`) |
| `configs.py` | CDNA3/MI300A-reasoned autotune config space + a problem-shape pruner |
| `test_correctness.py` | numerical check vs the oracle (GPU bf16 @ 2e-2, or CPU `TRITON_INTERPRET=1` fp32 @ 3e-3) |
| `benchmark.py` | microbenchmark over Kimi-K2.6 decode + prefill shapes; reports effective weight-read GB/s |

## Why Triton (and not Gluon yet)

tokenspeed's `AGENTS.md` prefers **Triton Gluon for AMD** and **Triton for
portable** kernels. This first cut is portable Triton so it (a) runs under
`TRITON_INTERPRET=1` for CPU correctness with no GPU, (b) compiles unchanged on
NVIDIA + AMD for parity, and (c) exposes the CDNA3 tuning knobs
(`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`) through the autotune `Config`.
A Gluon rewrite (explicit `BlockedLayout` + `amd_mfma` + LDS double-buffer,
mirroring tokenspeed's `ops/moe/gluon.py`) is the follow-up once the config space
below is validated on real gfx942 hardware.

## Key optimizations vs the in-tree kernel

The in-tree kernel (`tokenspeed .../ops/moe/triton.py`, `use_int4_w4a16` branch)
runs untuned with a single hardcoded small-M config and a **per-element** nibble
shift + per-element scale reload inside the K loop. This kernel:

1. **Autotunes** over a CDNA3-reasoned space keyed on the GEMM shape.
2. **Loads one int32 per 8 K and unpacks all 8 nibbles with a single broadcasted
   shift** over a length-8 constexpr vector — the unpack is amortized over 8 MACs
   and the weight read is one coalesced int32 load (4x fewer bytes than bf16).
3. **Reloads the group scale once per K-group, not per K-element**: with
   `BLOCK_K % group == 0` the scale tile is `[BLOCK_K // group, N]` and is
   broadcast across the 32 K of each group.
4. Picks **MFMA-friendly tiles**: `BLOCK_K` is a multiple of 8 (pack) and 32
   (group); `matrix_instr_nonkdim = 16` selects the 16x16 MFMA so tiny-M decode
   tiles do not waste MFMA M lanes; `waves_per_eu` is tuned per regime so small-M
   tiles raise occupancy to hide the int4 weight read while big prefill tiles stay
   low to avoid VGPR spills.

## Autotune config space (CDNA3 reasoning)

The decode regime (`M = 1 token x top_k -> tiny M`) is **weight-read-bandwidth
bound**: each expert weight is read once with almost no reuse, so the goal is to
maximize coalesced int4 weight-read bandwidth and minimize per-element dequant.
See `configs.py` for the full annotated list. Three regimes:

* **Decode / tiny-M** (`BLOCK_M in {16, 32}`): wide-ish `BLOCK_N` to amortize the
  per-tile scale fetch + launch overhead, high `waves_per_eu` (2-4) to hide the
  weight read, `num_warps` 2-4 (M too small to feed 8 wavefronts), `GROUP_SIZE_M
  = 1` (no L2 super-grouping — little weight reuse at tiny M).
* **Light prefill** (`M ~ 64-256`): balanced square-ish tiles.
* **Heavy prefill** (`M >= 512`): large `BLOCK_M`/`BLOCK_N`, more warps, low
  `waves_per_eu` (0-1) to avoid spilling the big accumulator, larger
  `GROUP_SIZE_M` for L2 reuse of the now-shared weights.

`BLOCK_K in {64, 128, 256}` (all multiples of pack=8 and group=32): 256 maximizes
weight-read coalescing but is paired with `num_stages <= 2` to fit the 64 KB LDS;
64 keeps register/LDS pressure low for the smallest tiles.

> These configs are **reasoned from the architecture, not yet measured on
> hardware** (no gfx942 available at authoring time). On-device autotune over
> this space is the immediate next step; the space is intentionally small enough
> to sweep quickly.

## Running

```bash
# correctness on GPU (NVIDIA or AMD gfx942), bf16, atol/rtol 2e-2
pytest int4_w4a16_moe/test_correctness.py

# correctness on CPU with no GPU (Triton interpreter), fp32, atol/rtol 3e-3
TRITON_INTERPRET=1 pytest int4_w4a16_moe/test_correctness.py

# microbenchmark over Kimi-K2.6 decode + prefill shapes (needs a GPU you hold;
# does NOT submit any cluster job)
python -m int4_w4a16_moe.benchmark
```

### Interpreter caveat

The Triton CPU interpreter (>= 3.4) mis-evaluates `tl.dot` with **bf16** operands
(returns garbage); the test therefore uses **fp32** activations under
`TRITON_INTERPRET=1`. Because `b_deq` is always cast to `a.dtype` before the dot,
the fp32 path exercises the identical kernel logic (unpack, scale broadcast, dot,
accumulate, masking, dispatch) — bf16 is validated on real hardware.
