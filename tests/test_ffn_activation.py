import torch

from xkernels.ops.ffn._activation import SwigluAct
from xkernels.utils.testing import assert_close


def _torch_swiglu(g, u):
    return torch.nn.functional.silu(g) * u


def test_forward_matches_torch():
    g = torch.randn(16, 16)
    u = torch.randn(16, 16)
    out = SwigluAct.apply(g, u, _torch_swiglu)
    assert_close(out, _torch_swiglu(g, u))


def test_backward_matches_autograd_reference():
    g = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    u = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    gr = g.detach().clone().requires_grad_(True)
    ur = u.detach().clone().requires_grad_(True)

    SwigluAct.apply(g, u, _torch_swiglu).sum().backward()
    _torch_swiglu(gr, ur).sum().backward()

    assert_close(g.grad, gr.grad)
    assert_close(u.grad, ur.grad)
