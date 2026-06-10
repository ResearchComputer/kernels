# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: fused residual-add + RMSNorm (issue #12) Triton kernel vs oracle.

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels.ops.comm.fused import add_rmsnorm_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


@pytest.mark.parametrize("T,H", [(4, 7168), (16, 512), (3, 320)])
def test_triton_matches_reference(T, H):
    dev = _device()
    try:
        from xkernels.ops.comm.triton.add_rmsnorm_kernel import add_rmsnorm_triton
    except Exception:
        pytest.skip("triton not available")
    dtype = torch.float32 if _INTERP else torch.bfloat16
    torch.manual_seed(0)
    x = (torch.randn(T, H, device=dev) * 0.5).to(dtype)
    residual = (torch.randn(T, H, device=dev) * 0.5).to(dtype)
    weight = (torch.randn(H, device=dev) * 0.1 + 1).to(dtype)

    out, new_res = add_rmsnorm_triton(x, residual, weight, 1e-6)
    ref_out, ref_res = add_rmsnorm_ref(x, residual, weight, 1e-6)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(new_res.float(), ref_res.float(), atol=atol, rtol=rtol)
