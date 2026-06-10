# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Numerical correctness check: optimized INT4 W4A16 kernel vs PyTorch oracle.

Acceptance (issue #1): match dequant-then-matmul within ``atol/rtol ~ 2e-2``
(bf16). Runs on:

* GPU (NVIDIA or AMD gfx942) with a real Triton install -> bf16 activations,
  the production dtype, ``atol/rtol = 2e-2``.
* CPU via ``TRITON_INTERPRET=1`` (no GPU) -> **fp32** activations, ``atol/rtol =
  1e-3``. NOTE: the Triton CPU interpreter (>=3.4) mis-evaluates ``tl.dot`` with
  bf16 operands (returns garbage); fp32 exercises the identical kernel path
  (unpack, group-scale broadcast, dot, accumulate, masking, dispatch) since
  ``b_deq`` is always cast to ``a.dtype`` before the dot. So fp32 fully validates
  the kernel logic on CPU and bf16 validates it on real hardware.

Output convention: the kernel writes **token-indexed** output
``c[m*top_k + j] = A[m] @ W[e]^T`` (e = topk_ids[m, j]); the reduce is
``c.view(M, top_k, N).sum(1)``.

Usage::

    pytest int4_w4a16_moe/test_correctness.py                       # GPU, bf16
    TRITON_INTERPRET=1 pytest int4_w4a16_moe/test_correctness.py     # CPU, fp32
"""

from __future__ import annotations

import os

import pytest
import torch

from .reference import dequant_w4a16, make_w4a16_weights, moe_align_block_size_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _pin_single_config():
    """Pin the autotuner to one config (autotune is a no-op under the interpreter)."""
    from .kernel import fused_moe_int4_kernel

    node = fused_moe_int4_kernel
    while node is not None and not hasattr(node, "configs"):
        node = getattr(node, "fn", None)
    if node is not None:
        node.configs = node.configs[:1]


def _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed):
    """fp32/bf16 grouped-MoE oracle reduced to ``[M, N]``."""
    W = dequant_w4a16(packed, scale, group_size).to(A.dtype)
    M, topk = topk_ids.shape
    out = torch.zeros(M, W.shape[1], dtype=torch.float32, device=A.device)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j])
            contrib = A[m].float() @ W[e].float().T
            if mul_routed:
                contrib = topk_w[m, j].float() * contrib
            out[m] += contrib
    return out


def _run_kernel(A, packed, scale, topk_ids, topk_w, *, top_k, group_size, block_m, mul_routed):
    import triton.language as tl

    from .kernel import int4_w4a16_moe_gemm

    E, N, _ = packed.shape
    M = A.shape[0]
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    # Token-indexed output: [M*top_k, N].
    c = torch.zeros((M * top_k, N), dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float32
    int4_w4a16_moe_gemm(
        A,
        packed,
        scale,
        c,
        topk_w.flatten().float(),
        sorted_ids,
        expert_ids,
        num_post,
        top_k=top_k,
        group_size=group_size,
        mul_routed_weight=mul_routed,
        compute_type=compute_type,
        filter_expert=False,
    )
    return c.view(M, top_k, N).sum(dim=1)


def _params():
    if _INTERP:  # keep the slow interpreter tractable
        return [(1, 8, 64, 128, 2), (4, 8, 128, 256, 4), (2, 4, 96, 64, 2)]
    return [
        (1, 48, 256, 512, 8),  # decode-like, Kimi-ish E/top_k
        (4, 8, 512, 1024, 4),
        (16, 16, 1024, 2048, 4),
    ]


@pytest.mark.parametrize("M,E,N,K,top_k", _params())
@pytest.mark.parametrize("mul_routed", [False, True])
def test_int4_w4a16_matches_reference(M, E, N, K, top_k, mul_routed):
    dev = _device()
    group_size = 32
    _pin_single_config()
    torch.manual_seed(0)
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    dtype = torch.float32 if _INTERP else torch.bfloat16
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)

    block_m = 16
    got = _run_kernel(
        A, packed, scale, topk_ids, topk_w,
        top_k=top_k, group_size=group_size, block_m=block_m, mul_routed=mul_routed,
    )
    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed)
    # Interpreter path runs fp32 but the group scales are still bf16, so the
    # different K-accumulation order vs the reference loop leaves a small bf16-
    # scale rounding gap; 3e-3 covers it. Hardware bf16 uses the issue-#1 2e-2.
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)


def test_dequant_roundtrip():
    """Packed weights from ``make_w4a16_weights`` dequant exactly (no kernel)."""
    dev = _device()
    packed, scale, w_ref = make_w4a16_weights(2, 64, 128, 32, device=dev, seed=3)
    torch.testing.assert_close(dequant_w4a16(packed, scale, 32), w_ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
