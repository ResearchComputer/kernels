# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark: moe_sum_reduce Triton kernel vs the torch oracle.

Kimi-K2.6 MoE geometry (top_k=8, hidden=7168) over a sweep of token counts.
Needs a GPU you already hold; does not submit a cluster job.

Usage::

    python benchmarks/bench_moe_sum_reduce.py
"""

from __future__ import annotations

import torch

from xkernels import moe_sum_reduce
from xkernels.ops.moe.sum_reduce import moe_sum_reduce_ref

TOP_K, H = 8, 7168  # Kimi-K2.6 top_k, hidden


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    dev = "cuda"
    print(f"{'M':>6} {'triton_ms':>10} {'torch_ms':>10} {'speedup':>8}")
    for M in [256, 1024, 4096, 16384]:
        y = torch.randn(M, TOP_K, H, device=dev, dtype=torch.bfloat16)
        w = torch.rand(M, TOP_K, device=dev, dtype=torch.float32)
        tri = triton.testing.do_bench(lambda y=y, w=w: moe_sum_reduce(y, w))
        ref = triton.testing.do_bench(lambda y=y, w=w: moe_sum_reduce_ref(y, w))
        print(f"{M:6d} {tri:10.4f} {ref:10.4f} {ref / tri:7.2f}x")


if __name__ == "__main__":
    main()
