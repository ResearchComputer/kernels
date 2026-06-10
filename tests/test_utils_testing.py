import pytest
import torch

from xkernels.utils.testing import assert_close, tolerance


def test_tolerance_known_dtypes():
    assert tolerance(torch.float32)["rtol"] < tolerance(torch.float16)["rtol"]
    assert "atol" in tolerance(torch.bfloat16)


def test_assert_close_passes_for_equal():
    a = torch.randn(8, 8)
    assert_close(a, a.clone())


def test_assert_close_raises_for_different():
    a = torch.zeros(4)
    b = torch.ones(4)
    with pytest.raises(AssertionError):
        assert_close(a, b)
