"""Public `moe` op (stub — dispatches to reference until kernels land)."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def moe(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_experts: torch.Tensor,
    *,
    top_k: int = 1,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Top-k MoE forward. See `moe_reference` for tensor shapes."""
    return dispatch("moe", x, w_gate, w_experts, top_k=top_k, backend=backend)
