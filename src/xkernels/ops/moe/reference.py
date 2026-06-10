"""Pure-torch reference MoE forward (test oracle / default backend).

TODO: add fused triton/cuda backends (grouped GEMM, token routing). The custom
backends should match this reference output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register


def moe_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_experts: torch.Tensor,
    top_k: int = 1,
) -> torch.Tensor:
    """Top-k softmax-routed MoE with per-expert linear maps.

    x: (T, d), w_gate: (d, E), w_experts: (E, d, d). Returns (T, d).
    """
    logits = x @ w_gate
    weights, idx = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    out = torch.zeros_like(x)
    for k in range(top_k):
        expert = idx[:, k]
        gate = weights[:, k].unsqueeze(-1)
        per_token = torch.einsum("td,tde->te", x, w_experts[expert])
        out = out + gate * per_token
    return out


register("moe", Backend.REFERENCE)(moe_reference)
