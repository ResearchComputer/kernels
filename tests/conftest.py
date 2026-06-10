"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture
def device() -> str:
    """The device tests should run kernels on ('cuda' if present, else 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"
