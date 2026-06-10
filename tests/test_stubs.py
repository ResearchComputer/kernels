import torch

from xkernels.ops.comm.reference import all_reduce_reference
from xkernels.ops.moe.reference import moe_reference


def test_moe_reference_runs():
    x = torch.randn(8, 16)
    w_gate = torch.randn(16, 4)  # 4 experts
    w_experts = torch.randn(4, 16, 16)
    out = moe_reference(x, w_gate, w_experts, top_k=1)
    assert out.shape == (8, 16)


def test_comm_reference_is_identity_single_process():
    x = torch.randn(4, 4)
    torch.testing.assert_close(all_reduce_reference([x])[0], x)
