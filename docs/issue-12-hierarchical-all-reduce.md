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

## Open: graph-captured measurement

The decisive decode-regime number needs the collective sequence captured in a
HIP graph. Capturing the **networked RCCL/OFI-CXI** collectives is blocked on this
stack (PyTorch 2.11+rocm7.2, RCCL 2.27.7, from-source aws-ofi-rccl):

- The PG watchdog calls `hipEventQuery` on the capturing stream → `operation not
  permitted when stream is capturing` (abort).
- Disabling the watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=0`) then surfaces an
  OFI memory-registration `invalid device pointer`.

So the graph-captured win is best measured in the **serve's own HIP-graph capture
path** (tokenspeed-amd), which captures the whole decode step including collectives
and handles the watchdog. Tracked as the remaining work on #12.

## Environment note (for whoever maintains the EDFs)

`~/.edf/tokenspeed-rocm-aiter-myofi.toml` no longer sets `LD_LIBRARY_PATH` to the
from-source plugin at
`/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib`.
Without it RCCL can't find `librccl-net-ofi.so` and fails with *“Failed to
initialize any NET plugin.”* `slurm/bench_allreduce_beverin.sbatch` sets it
explicitly; consider restoring it in the EDF `[env]`.
