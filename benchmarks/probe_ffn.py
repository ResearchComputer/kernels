# SPDX-License-Identifier: MIT
"""Diagnose FFN GEMM speed on gfx942: is the torch bf16 matmul the bottleneck?"""
from __future__ import annotations

import torch


def _ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    import torch.nn.functional as F

    dev = "cuda"
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    flop = 2 * 4096 * 4096 * 11008
    for dt in (torch.bfloat16, torch.float16):
        a = torch.randn(4096, 4096, device=dev, dtype=dt)
        b = torch.randn(4096, 11008, device=dev, dtype=dt)
        ms = _ms(lambda a=a, b=b: a @ b)
        print(f"  matmul [4096,4096]x[4096,11008] {str(dt):16}: "
              f"{ms:8.3f} ms  {flop / ms / 1e9:7.1f} TFLOP/s")
    # full reference FFN at two sizes (bf16 — exposes the slow GEMM path)
    for M in (512, 4096):
        x = torch.randn(M, 4096, device=dev, dtype=torch.bfloat16)
        wg = torch.randn(4096, 11008, device=dev, dtype=torch.bfloat16)
        wu = torch.randn(4096, 11008, device=dev, dtype=torch.bfloat16)
        wd = torch.randn(11008, 4096, device=dev, dtype=torch.bfloat16)
        ms = _ms(lambda x=x, wg=wg, wu=wu, wd=wd: (F.silu(x @ wg) * (x @ wu)) @ wd)
        print(f"  ffn_reference M={M:5d}: {ms:8.3f} ms")


if __name__ == "__main__":
    main()
