"""Normalization kernels.

Ships the fused parallel dual RMSNorm (MLA ``q_a`` / ``kv_a`` latents, issue #2):
two independent RMSNorms over differently-sized feature dims in a single launch.
"""
from .interface import dual_rmsnorm

# Import the Triton backend for its registration side effect. Optional.
try:  # pragma: no cover - requires triton
    from .triton import dual_rmsnorm_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["dual_rmsnorm"]
