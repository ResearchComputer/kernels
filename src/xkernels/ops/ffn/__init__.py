"""Fused FFN kernels."""
from .interface import fused_ffn

# Import backend modules for their registration side effects. Triton/CUDA are
# optional — guard so the package imports on any machine.
try:  # pragma: no cover - hardware dependent
    from .triton import ffn_kernel  # noqa: F401
except Exception:
    pass

try:  # pragma: no cover - requires compiled extension
    from . import cuda  # noqa: F401
except Exception:
    pass

__all__ = ["fused_ffn"]
