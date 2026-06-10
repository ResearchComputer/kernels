# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""INT4 W4A16 grouped fused-MoE GEMM for AMD MI300A (gfx942). See issue #1."""

from .configs import get_autotune_configs, prune_configs
from .reference import (
    dequant_w4a16,
    make_w4a16_weights,
    moe_align_block_size_ref,
    moe_w4a16_ref,
)

__all__ = [
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
    "moe_w4a16_ref",
    "get_autotune_configs",
    "prune_configs",
]

try:  # kernel import requires triton; reference/baseline does not.
    from .kernel import fused_moe_int4_kernel, int4_w4a16_moe_gemm  # noqa: F401

    __all__ += ["fused_moe_int4_kernel", "int4_w4a16_moe_gemm"]
except ImportError:  # pragma: no cover - allow baseline-only environments
    pass
