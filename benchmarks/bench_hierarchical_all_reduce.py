# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Distributed correctness + latency eval for the hierarchical all-reduce (issue #12).

Launch with torchrun (local CPU/gloo smoke) or SLURM srun (real RCCL on MI300A):

    # local logical 8-rank smoke (2 nodes x 4), gloo/CPU, fp32:
    torchrun --nproc-per-node=8 benchmarks/bench_hierarchical_all_reduce.py --ranks-per-node 4

    # on beverin via srun (see slurm/bench_allreduce_beverin.sbatch), nccl/RCCL, bf16

Reads rank/world from torchrun (RANK/WORLD_SIZE/LOCAL_RANK) or SLURM
(SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID). Backend is nccl when CUDA/ROCm is
visible, else gloo. Checks ``hierarchical_all_reduce == flat_all_reduce`` and
reports per-collective latency (flat vs hierarchical) at decode sizes.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from xkernels.ops.comm import (
    build_topology_groups,
    flat_all_reduce,
    hierarchical_all_reduce,
)


def _env(*names, default=0):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def _init():
    rank = _env("RANK", "SLURM_PROCID")
    world = _env("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _env("LOCAL_RANK", "SLURM_LOCALID")
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    return rank, world, local_rank, device, use_cuda


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    dist.barrier()
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--dtype", default=None, help="bf16|float32 (default: bf16 on GPU, float32 CPU)"
    )
    args = ap.parse_args()

    rank, world, local_rank, device, use_cuda = _init()
    dtype = (
        getattr(torch, args.dtype)
        if args.dtype
        else (torch.bfloat16 if use_cuda else torch.float32)
    )
    intra, cross, info = build_topology_groups(args.ranks_per_node, world)
    if rank == 0:
        print(
            f"world={world} ranks_per_node={info.ranks_per_node} num_nodes={info.num_nodes} "
            f"backend={'nccl' if use_cuda else 'gloo'} dtype={dtype}",
            flush=True,
        )

    # ---- correctness: hierarchical == flat ----
    ok = True
    for bs in args.sizes:
        torch.manual_seed(1000 + rank)  # distinct data per rank
        x = (torch.randn(bs, args.hidden, device=device) * 0.1).to(dtype)
        flat = flat_all_reduce(x, group=None)
        hier = hierarchical_all_reduce(x, intra, cross)
        atol = rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        close = torch.allclose(hier.float(), flat.float(), atol=atol, rtol=rtol)
        ok = ok and close
        if rank == 0:
            print(f"  correctness bs={bs:>3}: {'OK' if close else 'MISMATCH'}", flush=True)
    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"correctness (all ranks): {'PASS' if flag.item() else 'FAIL'}", flush=True)

    # ---- latency: flat vs hierarchical ----
    if rank == 0:
        print(f"\n{'bs':>4} {'flat_ms':>10} {'hier_ms':>10} {'speedup':>8}", flush=True)
    for bs in args.sizes:
        x = (torch.randn(bs, args.hidden, device=device) * 0.1).to(dtype)
        flat_ms = _bench(
            lambda x=x: flat_all_reduce(x, group=None), device, args.iters, args.warmup
        )
        hier_ms = _bench(
            lambda x=x: hierarchical_all_reduce(x, intra, cross),
            device,
            args.iters,
            args.warmup,
        )
        if rank == 0:
            print(
                f"{bs:>4} {flat_ms:>10.4f} {hier_ms:>10.4f} {flat_ms / hier_ms:>7.2f}x",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
