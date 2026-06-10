import pytest

from xkernels._backends import Backend
from xkernels._dispatch import dispatch, register, registered_backends


def setup_function():
    # Register a couple of fake backends for an isolated kernel name.
    @register("_unit", Backend.REFERENCE)
    def _ref(x):
        return ("reference", x)

    @register("_unit", Backend.TRITON)
    def _triton(x):
        return ("triton", x)


def test_registered_backends_lists_what_was_registered():
    assert set(registered_backends("_unit")) >= {Backend.REFERENCE, Backend.TRITON}


def test_explicit_backend_is_honored():
    assert dispatch("_unit", 5, backend=Backend.TRITON)[0] == "triton"


def test_string_backend_is_accepted():
    assert dispatch("_unit", 5, backend="reference")[0] == "reference"


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        dispatch("_unit", 5, backend=Backend.CUDA)


def test_unknown_kernel_raises():
    with pytest.raises(KeyError):
        dispatch("_does_not_exist", 5)
