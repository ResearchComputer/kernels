import torch

from xkernels._backends import Backend, detect_vendor, device_of


def test_backend_enum_has_expected_members():
    assert {b.name for b in Backend} >= {"TRITON", "CUDA", "HIP", "REFERENCE"}


def test_detect_vendor_returns_known_value():
    assert detect_vendor() in {"nvidia", "amd", "none"}


def test_device_of_cpu_tensor():
    assert device_of(torch.zeros(2)) == "cpu"
