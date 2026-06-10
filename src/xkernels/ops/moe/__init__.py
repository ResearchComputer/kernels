"""Mixture-of-Experts kernels.

Ships the INT4 W4A16 grouped fused-MoE GEMM (issue #1) and the weighted top-k
reduction (issue #5). Each public op dispatches across a pure-torch reference
(default on CPU / no Triton) and an autotuned Triton backend.
"""
from .interface import fused_moe_int4_w4a16
from .sum_reduce import moe_sum_reduce
from .w4a16 import dequant_w4a16, make_w4a16_weights, moe_align_block_size_ref

# Import Triton backends for their registration side effects. Optional — guard
# each so the package imports without Triton installed.
try:  # pragma: no cover - requires triton
    from .triton import moe_int4_kernel  # noqa: F401
except Exception:
    pass

try:  # pragma: no cover - requires triton
    from .triton import sum_reduce_kernel  # noqa: F401
except Exception:
    pass

__all__ = [
    "fused_moe_int4_w4a16",
    "moe_sum_reduce",
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
]
