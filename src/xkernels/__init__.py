"""xkernels — customized compute kernels across vendors and kernel types."""

from .ops.ffn import fused_ffn

__version__ = "0.0.1"
__all__ = ["fused_ffn", "__version__"]
