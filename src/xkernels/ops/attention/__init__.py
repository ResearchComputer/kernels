"""Attention kernels.

Ships ``mha_merge_state`` (issue #3): the numerically-stable online-softmax
merge of two attention partials by their log-sum-exp, used by chunked-prefill /
split-KV MLA on AMD MI300A.
"""
from .interface import mha_merge_state

# Import the Triton backend for its registration side effect. Optional.
try:  # pragma: no cover - requires triton
    from .triton import merge_state_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["mha_merge_state"]
