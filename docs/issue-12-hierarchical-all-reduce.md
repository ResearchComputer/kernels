# Issue #12 — Topology-aware hierarchical all-reduce: design + on-cluster results

## What landed

`xkernels.ops.comm`, a distributed-collective module (not a single-GPU dispatched
kernel — collectives take process groups):

- `build_topology_groups(ranks_per_node)` — intra-node (xGMI) + cross-node (CXI,
  same-local-rank) process groups, contiguous block layout.
- `hierarchical_all_reduce(x, intra, cross)` — `reduce_scatter` (xGMI) → cross-node
  `all_reduce` of the 1/rpp partial (CXI) → `all_gather` (xGMI).
- `flat_all_reduce(x, group)` — the oracle.
- `residual_rmsnorm` + Triton `add_rmsnorm` kernel — the fused residual-add+RMSNorm
  epilogue, and `hierarchical_all_reduce_residual_rmsnorm` composing the two.

## Validation

| Check | Where | Result |
|-------|-------|--------|
| Schedule correctness, 8-rank logical (2×4) | local gloo/CPU | **PASS** all sizes |
| Fused residual+RMSNorm vs torch oracle | `TRITON_INTERPRET=1` | **PASS** |
| Correctness, single-node 4-rank | beverin MI300A, RCCL | **PASS** (bf16) |
| Correctness, **2-node 8-rank over CXI** | beverin, `myofi` + OFI plugin | **PASS** bs∈{1,2,4,8,16} |
| RCCL uses CXI fabric | beverin | `NET/OFI Selected provider is cxi … (found 4 nics)` |

`hierarchical_all_reduce` is numerically equal to `flat_all_reduce` within bf16
tolerance on real CXI — acceptance #1 met.

## Latency finding (2-node MI300A, eager, job 380669)

```
 bs    flat_ms    hier_ms  speedup
  1     0.0953     0.2106    0.45x
  2     0.1068     0.2209    0.48x
  4     0.1079     0.2187    0.49x
  8     0.0993     0.2170    0.46x
 16     0.1072     0.2172    0.49x
```

**Eager, the hierarchical schedule loses (~0.47×) at decode sizes.** At 14 KiB
(`bs×7168` bf16) the collective is *launch-latency-bound*, and the hierarchical
schedule issues **3 collectives** (reduce-scatter + cross all-reduce + all-gather)
vs flat's **1** — 3× the per-launch overhead swamps the reduced cross-node payload.
A flat 8-rank RCCL all-reduce of 14 KiB over CXI is already ~0.10 ms.

This is consistent with the issue's premise: the win exists only when the
per-launch cost is amortized, i.e. **under HIP-graph decode capture** (the serve's
regime). It is not visible in an eager microbenchmark.

## Graph-captured measurement — cracked

Capturing the **networked RCCL/OFI-CXI** collectives in a HIP graph was blocked by
*two independent* failures on this stack (PyTorch 2.11+rocm7.2, RCCL 2.27.7,
from-source aws-ofi-rccl). Both are now solved:

1. **OFI memory-registration** `invalid device pointer` mid-capture. Fixed by
   making CXI MR registration capture-safe: `FI_CXI_OPTIMIZED_MRS=0` (or the
   libfabric MR cache via `FI_MR_CACHE_MONITOR=memhooks` +
   `NCCL_OFI_MR_CACHE_DISABLE=0`).
2. **PG watchdog** calling `hipEventQuery` on the capturing stream → `operation
   not permitted when stream is capturing`. *Not* env-tunable here
   (`TORCH_NCCL_ASYNC_ERROR_HANDLING=0` does not stop the thread). Fixed by
   driving the collectives through a **watchdog-free raw RCCL communicator**
   (`benchmarks/pynccl_lite.py`, ctypes over `librccl.so`) — the same approach
   vLLM/SGLang use. The unique-id handshake rides the existing torch.distributed
   store; no collective ever goes through the watchdog'd PG.

`benchmarks/bench_capture_pynccl.py` (driver: `slurm/bench_capture_beverin.sbatch`)
captures both schedules and replays them. Correctness re-validated through the raw
comm (flat sum == world, hier == flat) before timing.

## Captured latency finding (2-node MI300A, HIP graph, job 380930)

```
   bs       MB    flat_ms    hier_ms  speedup
    1    0.014     0.0682     0.0939    0.73x
    2    0.029     0.0776     0.1030    0.75x
    4    0.057     0.1123     0.1012    1.11x
    8    0.115     0.1091     0.1079    1.01x
   16    0.229     0.1126     0.1177    0.96x
   64    0.918     0.1372     0.1383    0.99x
  256    3.670     0.1874     0.1759    1.07x
 1024   14.680     0.3291     0.3719    0.88x
 4096   58.720     0.8714     1.1926    0.73x
```

**Capture does what we predicted — it amortizes the per-launch penalty.** The
decode (bs=1) flat-vs-hier ratio improved from eager **0.45×** to captured
**0.73×**: with the 3 collectives' launch overhead removed from the graph, most of
the gap closes. That was the technical claim the deep-dive set out to prove, and it
holds.

**But hierarchical still does not beat flat — at any size, in either mode.** The
crossover never arrives. The reason is the baseline: on this stack RCCL's *flat*
all-reduce is **already topology-aware** — it discovers the 4 CXI NICs, builds NIC
groups, and runs an internally hierarchical schedule (`Selected provider is cxi …
found 4 nics`, NIC groups 0–3, `SENDRECV`). Our hand-written
`reduce_scatter`(xGMI) → `all_reduce`(CXI, ¼ payload) → `all_gather`(xGMI) is doing
manually what RCCL already does inside one launch, so:

- **Decode (≤14 KiB, bs≤2):** latency-bound. Even captured, 3 serial dependent
  graph nodes (each with its own intra-/cross-node sync) cost more than flat's one
  → **0.73–0.75×**.
- **Mid (bs 4–256, 57 KiB–3.7 MB):** roughly even, occasionally +7–11% — within
  run-to-run noise, no robust win.
- **Bandwidth (bs≥1024, ≥14.7 MB):** the extra xGMI `reduce_scatter`+`all_gather`
  passes are pure overhead on top of an already-NIC-saturating flat collective →
  **0.73× at 58.7 MB.** No crossover.

**Conclusion for #12.** The fused-epilogue + hierarchical schedule is *correct* on
real 2-node CXI (acceptance #1 met), and the graph-capture blocker is fully cracked
and documented. The *performance* premise — that a hand-rolled hierarchical
decomposition beats the oracle at decode — does **not** hold on this 2-node /
4-NIC-per-node MI300A stack, because the RCCL/OFI flat all-reduce is already
hierarchical internally. A manual decomposition would be expected to pay off only
where the vendor collective is *not* topology-aware, or at larger node counts /
different NIC ratios where the cross-node payload reduction outweighs the extra
intra-node passes. Capture amortization is real (0.45×→0.73×) but insufficient to
cross 1.0× here.

## Environment note (for whoever maintains the EDFs)

`~/.edf/tokenspeed-rocm-aiter-myofi.toml` no longer sets `LD_LIBRARY_PATH` to the
from-source plugin at
`/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib`.
Without it RCCL can't find `librccl-net-ofi.so` and fails with *“Failed to
initialize any NET plugin.”* `slurm/bench_allreduce_beverin.sbatch` sets it
explicitly; consider restoring it in the EDF `[env]`.
