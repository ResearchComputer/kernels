# xkernels Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `xkernels` repository — a PyTorch kernel library organized kernel-type-first with pluggable per-vendor backends (Triton / CUDA-HIP / reference), plus a correctness-test and benchmark harness, with FFN fully worked and MoE/comm stubbed.

**Architecture:** Kernel **type** is the top axis under `src/xkernels/ops/`. Each type exposes one public op that dispatches to a registered backend selected from the runtime device/vendor, falling back to a pure-torch reference. The worked example is a **SwiGLU FFN** — `(silu(x @ w_gate) * (x @ w_up)) @ w_down` — where the projection matmuls use `torch.matmul` and the custom Triton/CUDA kernel fuses the elementwise `silu(g) * u` activation. The custom activation is wrapped in `torch.autograd.Function` (torch-computed backward) so the op is end-to-end differentiable.

**Tech Stack:** Python 3.10+, PyTorch, Triton, CUDA/HIP via `torch.utils.cpp_extension`, pytest, ruff, GitHub Actions.

**Reference spec:** `docs/superpowers/specs/2026-06-10-xkernels-scaffold-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata, deps, optional-dep groups, ruff/pytest config |
| `setup.py` | Discovers `ops/*/cuda/*.cu` and builds optional CUDA/HIP extensions |
| `LICENSE` | MIT license text |
| `.gitignore` | Python / build-artifact ignores |
| `README.md` | Overview, install, usage, layout |
| `.pre-commit-config.yaml` | ruff lint + format hooks |
| `.github/workflows/ci.yml` | Lint + CPU/reference test job |
| `src/xkernels/__init__.py` | Public re-exports + `__version__` |
| `src/xkernels/_backends.py` | `Backend` enum + vendor/device detection |
| `src/xkernels/_dispatch.py` | Registry (`register`) + selection (`dispatch`) |
| `src/xkernels/utils/testing.py` | `assert_close` + per-dtype tolerance presets |
| `src/xkernels/utils/benchmarking.py` | `benchmark()` timing helper |
| `src/xkernels/ops/ffn/reference.py` | Pure-torch SwiGLU FFN (test oracle) |
| `src/xkernels/ops/ffn/_activation.py` | Shared `SwigluAct` autograd.Function |
| `src/xkernels/ops/ffn/interface.py` | `fused_ffn` public op + dispatch call |
| `src/xkernels/ops/ffn/triton/ffn_kernel.py` | Triton fused-activation backend |
| `src/xkernels/ops/ffn/cuda/ffn.cu` | CUDA/HIP fused-activation kernel + pybind |
| `src/xkernels/ops/ffn/cuda/__init__.py` | Loads compiled ext, registers CUDA/HIP backend |
| `src/xkernels/ops/{moe,comm}/{interface,reference}.py` | Stubs |
| `tests/conftest.py` | Device + available-backend fixtures |
| `tests/test_*.py` | Per-component tests |
| `benchmarks/bench_ffn.py` | Shape × backend sweep |
| `examples/ffn_usage.py` | Minimal usage example |
| `docs/adding-a-kernel.md` | Extension guide |

---

## Task 1: Repo skeleton + installable empty package

**Files:**
- Create: `.gitignore`, `LICENSE`, `pyproject.toml`, `setup.py`, `src/xkernels/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.so
*.egg-info/
build/
dist/
.eggs/
.pytest_cache/
.ruff_cache/
.venv/
venv/
*.csv
*.pdf
.DS_Store
```

- [ ] **Step 2: Create `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 Xiaozhe Yao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch"]
build-backend = "setuptools.build_meta"

[project]
name = "xkernels"
version = "0.0.1"
description = "Customized compute kernels across hardware vendors and kernel types"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [{ name = "Xiaozhe Yao" }]
dependencies = ["torch>=2.1"]

[project.optional-dependencies]
triton = ["triton>=2.1"]
bench = ["pandas", "matplotlib"]
dev = ["pytest>=7", "ruff>=0.4", "pre-commit"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
xkernels = ["ops/**/*.cu", "ops/**/*.cpp", "ops/**/*.h"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests", "benchmarks"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 4: Create `setup.py` (optional CUDA/HIP extensions)**

```python
"""Build script. Pure-Python install works without a CUDA/ROCm toolkit;
compiled extensions are added opportunistically, one per kernel type that
ships a `cuda/` directory."""
import glob
import os

from setuptools import setup

ext_modules = []
cmdclass = {}

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or os.environ.get("XKERNELS_FORCE_BUILD") == "1":
        for cu_dir in sorted(glob.glob("src/xkernels/ops/*/cuda")):
            kernel_type = cu_dir.split(os.sep)[-2]
            sources = sorted(glob.glob(os.path.join(cu_dir, "*.cu")))
            if not sources:
                continue
            ext_modules.append(
                CUDAExtension(
                    name=f"xkernels.ops.{kernel_type}.cuda._cuda",
                    sources=sources,
                )
            )
        if ext_modules:
            cmdclass["build_ext"] = BuildExtension
except Exception as exc:  # torch missing or build env broken — ship pure Python
    print(f"[xkernels setup] skipping compiled extensions: {exc}")

setup(ext_modules=ext_modules, cmdclass=cmdclass)
```

- [ ] **Step 5: Create `src/xkernels/__init__.py` (minimal, expands in later tasks)**

```python
"""xkernels — customized compute kernels across vendors and kernel types."""

__version__ = "0.0.1"
```

- [ ] **Step 6: Install and verify import**

Run: `pip install -e . && python -c "import xkernels; print(xkernels.__version__)"`
Expected: prints `0.0.1` (CUDA ext may build or be skipped — both are fine).

- [ ] **Step 7: Commit**

```bash
git add .gitignore LICENSE pyproject.toml setup.py src/xkernels/__init__.py
git commit -m "feat: installable empty xkernels package skeleton"
```

---

## Task 2: Backend detection

**Files:**
- Create: `src/xkernels/_backends.py`, `tests/test_backends.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends.py`:

```python
import torch

from xkernels._backends import Backend, detect_vendor, device_of


def test_backend_enum_has_expected_members():
    assert {b.name for b in Backend} >= {"TRITON", "CUDA", "HIP", "REFERENCE"}


def test_detect_vendor_returns_known_value():
    assert detect_vendor() in {"nvidia", "amd", "none"}


def test_device_of_cpu_tensor():
    assert device_of(torch.zeros(2)) == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xkernels._backends'`

- [ ] **Step 3: Write the implementation**

Create `src/xkernels/_backends.py`:

```python
"""Backend identifiers and hardware/vendor detection."""
from __future__ import annotations

import enum

import torch


class Backend(enum.Enum):
    REFERENCE = "reference"
    TRITON = "triton"
    CUDA = "cuda"
    HIP = "hip"


def detect_vendor() -> str:
    """Return the GPU vendor of the current torch build: 'nvidia', 'amd', or 'none'."""
    if getattr(torch.version, "hip", None):
        return "amd"
    if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
        return "nvidia"
    return "none"


def device_of(tensor: torch.Tensor) -> str:
    """Return 'cpu' or 'cuda' for a tensor's device type (ROCm reports 'cuda')."""
    return tensor.device.type
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/_backends.py tests/test_backends.py
git commit -m "feat: backend enum and vendor/device detection"
```

---

## Task 3: Dispatch registry

**Files:**
- Create: `src/xkernels/_dispatch.py`, `tests/test_dispatch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dispatch.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dispatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xkernels._dispatch'`

- [ ] **Step 3: Write the implementation**

Create `src/xkernels/_dispatch.py`:

```python
"""Backend registry and selection.

Backends self-register with `@register(kernel_name, Backend.X)`. The public op
calls `dispatch(kernel_name, *args, backend="auto", **kwargs)`, which resolves:
explicit arg -> env override (XKERNELS_BACKEND) -> auto (per-vendor preference),
falling back to REFERENCE.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from ._backends import Backend, detect_vendor

# kernel_name -> {Backend: callable}
_REGISTRY: dict[str, dict[Backend, Callable]] = {}

# Per-vendor preference order for "auto" selection (first available wins).
_AUTO_ORDER: dict[str, list[Backend]] = {
    "nvidia": [Backend.CUDA, Backend.TRITON, Backend.REFERENCE],
    "amd": [Backend.HIP, Backend.TRITON, Backend.REFERENCE],
    "none": [Backend.REFERENCE],
}


def register(kernel_name: str, backend: Backend) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _REGISTRY.setdefault(kernel_name, {})[backend] = fn
        return fn

    return deco


def registered_backends(kernel_name: str) -> list[Backend]:
    return list(_REGISTRY.get(kernel_name, {}).keys())


def _coerce(backend: Backend | str) -> Backend:
    return backend if isinstance(backend, Backend) else Backend(backend)


def select_backend(kernel_name: str, backend: Backend | str = "auto") -> Backend:
    if kernel_name not in _REGISTRY:
        raise KeyError(f"no backends registered for kernel '{kernel_name}'")
    impls = _REGISTRY[kernel_name]

    if backend != "auto":
        chosen = _coerce(backend)
        if chosen not in impls:
            raise KeyError(
                f"backend {chosen.name} not registered for '{kernel_name}'; "
                f"have {[b.name for b in impls]}"
            )
        return chosen

    env = os.environ.get("XKERNELS_BACKEND")
    if env:
        chosen = Backend(env.lower())
        if chosen in impls:
            return chosen

    for candidate in _AUTO_ORDER.get(detect_vendor(), [Backend.REFERENCE]):
        if candidate in impls:
            return candidate
    # Last resort: anything registered.
    return next(iter(impls))


def dispatch(kernel_name: str, *args, backend: Backend | str = "auto", **kwargs):
    chosen = select_backend(kernel_name, backend)
    return _REGISTRY[kernel_name][chosen](*args, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dispatch.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/_dispatch.py tests/test_dispatch.py
git commit -m "feat: backend registry and dispatch selection"
```

---

## Task 4: Testing & benchmarking utilities

**Files:**
- Create: `src/xkernels/utils/__init__.py`, `src/xkernels/utils/testing.py`, `src/xkernels/utils/benchmarking.py`, `tests/test_utils_testing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils_testing.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils_testing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xkernels.utils'`

- [ ] **Step 3: Write the implementations**

Create `src/xkernels/utils/__init__.py`:

```python
"""Shared helpers for tests and benchmarks."""
```

Create `src/xkernels/utils/testing.py`:

```python
"""Correctness-test helpers with per-dtype tolerance presets."""
from __future__ import annotations

import torch

_TOL: dict[torch.dtype, dict[str, float]] = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
}


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    """Return {'rtol', 'atol'} for a dtype (defaults to float32 tolerances)."""
    return _TOL.get(dtype, _TOL[torch.float32])


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """torch.testing.assert_close with tolerances keyed off `expected.dtype`."""
    tol = tolerance(expected.dtype)
    torch.testing.assert_close(actual, expected, rtol=tol["rtol"], atol=tol["atol"])
```

Create `src/xkernels/utils/benchmarking.py`:

```python
"""Timing helpers. Uses Triton's do_bench when available, else CUDA events."""
from __future__ import annotations

from collections.abc import Callable

import torch


def benchmark(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> float:
    """Return median wall-clock milliseconds per call of `fn`."""
    try:
        from triton.testing import do_bench

        return float(do_bench(fn, warmup=warmup, rep=iters))
    except Exception:
        pass

    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    import time

    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils_testing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/utils tests/test_utils_testing.py
git commit -m "feat: testing tolerance presets and benchmark timing helper"
```

---

## Task 5: FFN reference backend + public op

**Files:**
- Create: `src/xkernels/ops/__init__.py`, `src/xkernels/ops/ffn/__init__.py`, `src/xkernels/ops/ffn/reference.py`, `src/xkernels/ops/ffn/interface.py`, `tests/test_ffn.py`
- Modify: `src/xkernels/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ffn.py`:

```python
import torch

from xkernels import fused_ffn
from xkernels._backends import Backend
from xkernels.ops.ffn.reference import ffn_reference
from xkernels.utils.testing import assert_close


def _inputs(M=16, d_model=32, d_ff=64, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, d_model, dtype=dtype, generator=g)
    w_gate = torch.randn(d_model, d_ff, dtype=dtype, generator=g)
    w_up = torch.randn(d_model, d_ff, dtype=dtype, generator=g)
    w_down = torch.randn(d_ff, d_model, dtype=dtype, generator=g)
    return x, w_gate, w_up, w_down


def test_reference_matches_manual_swiglu():
    x, wg, wu, wd = _inputs()
    expected = (torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd
    assert_close(ffn_reference(x, wg, wu, wd), expected)


def test_public_op_reference_backend_on_cpu():
    x, wg, wu, wd = _inputs()
    out = fused_ffn(x, wg, wu, wd, backend=Backend.REFERENCE)
    assert out.shape == (x.shape[0], wd.shape[1])
    assert_close(out, ffn_reference(x, wg, wu, wd))


def test_public_op_preserves_leading_dims():
    x = torch.randn(3, 5, 32)
    wg = torch.randn(32, 64)
    wu = torch.randn(32, 64)
    wd = torch.randn(64, 32)
    out = fused_ffn(x, wg, wu, wd, backend=Backend.REFERENCE)
    assert out.shape == (3, 5, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ffn.py -v`
Expected: FAIL with `ImportError: cannot import name 'fused_ffn' from 'xkernels'`

- [ ] **Step 3: Write the reference backend**

Create `src/xkernels/ops/__init__.py`:

```python
"""Kernel implementations, organized by kernel type."""
```

Create `src/xkernels/ops/ffn/reference.py`:

```python
"""Pure-torch SwiGLU FFN — the correctness oracle and default backend."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register


def ffn_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Compute (silu(x @ w_gate) * (x @ w_up)) @ w_down."""
    return (F.silu(x @ w_gate) * (x @ w_up)) @ w_down


register("ffn", Backend.REFERENCE)(ffn_reference)
```

- [ ] **Step 4: Write the interface (public op + shape handling)**

Create `src/xkernels/ops/ffn/interface.py`:

```python
"""Public `fused_ffn` op: normalizes leading dims, then dispatches."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def fused_ffn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """SwiGLU FFN: (silu(x @ w_gate) * (x @ w_up)) @ w_down.

    `x` may have any number of leading dims; only the last (feature) dim must
    match `w_gate`/`w_up`. `backend` is "auto" or a `Backend` / its string value.
    """
    *lead, d_model = x.shape
    x2d = x.reshape(-1, d_model)
    out = dispatch("ffn", x2d, w_gate, w_up, w_down, backend=backend)
    return out.reshape(*lead, out.shape[-1])
```

Create `src/xkernels/ops/ffn/__init__.py`:

```python
"""Fused FFN kernels."""
from .interface import fused_ffn

# Import backend modules for their registration side effects. Triton/CUDA are
# optional — guard so the package imports on any machine.
try:  # pragma: no cover - hardware dependent
    from .triton import ffn_kernel  # noqa: F401
except Exception:
    pass

try:  # pragma: no cover - requires compiled extension
    from . import cuda  # noqa: F401
except Exception:
    pass

__all__ = ["fused_ffn"]
```

- [ ] **Step 5: Re-export from the package root**

Replace `src/xkernels/__init__.py` with:

```python
"""xkernels — customized compute kernels across vendors and kernel types."""

from .ops.ffn import fused_ffn

__version__ = "0.0.1"
__all__ = ["fused_ffn", "__version__"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_ffn.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add src/xkernels/ops src/xkernels/__init__.py tests/test_ffn.py
git commit -m "feat: FFN reference backend and public fused_ffn op"
```

---

## Task 6: Shared SwiGLU activation autograd.Function

**Files:**
- Create: `src/xkernels/ops/ffn/_activation.py`, `tests/test_ffn_activation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ffn_activation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ffn_activation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xkernels.ops.ffn._activation'`

- [ ] **Step 3: Write the implementation**

Create `src/xkernels/ops/ffn/_activation.py`:

```python
"""Autograd wrapper for the fused SwiGLU activation `silu(g) * u`.

Forward runs a backend-provided elementwise kernel; backward is computed in
torch so every backend (triton/cuda) is differentiable without a custom
backward kernel. d/dg[silu(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g))).
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class SwigluAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, kernel: Callable):
        ctx.save_for_backward(g, u)
        return kernel(g, u)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g, u = ctx.saved_tensors
        sig = torch.sigmoid(g)
        silu = g * sig
        dsilu_dg = sig * (1 + g * (1 - sig))
        grad_g = grad_out * u * dsilu_dg
        grad_u = grad_out * silu
        return grad_g, grad_u, None  # None for the non-tensor `kernel` arg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ffn_activation.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/ffn/_activation.py tests/test_ffn_activation.py
git commit -m "feat: shared SwiGLU activation autograd.Function"
```

---

## Task 7: FFN Triton backend

**Files:**
- Create: `src/xkernels/ops/ffn/triton/__init__.py`, `src/xkernels/ops/ffn/triton/ffn_kernel.py`
- Modify: `tests/test_ffn.py` (add a backend-parametrized test)

- [ ] **Step 1: Write the failing test (append to `tests/test_ffn.py`)**

Add to the bottom of `tests/test_ffn.py`:

```python
import pytest

from xkernels._dispatch import registered_backends

_GPU_BACKENDS = [
    b for b in registered_backends("ffn") if b not in (Backend.REFERENCE,)
]


@pytest.mark.parametrize("backend", _GPU_BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gpu_backend_matches_reference(backend, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
    x, wg, wu, wd = _inputs(dtype=dtype)
    x, wg, wu, wd = (t.cuda() for t in (x, wg, wu, wd))
    out = fused_ffn(x, wg, wu, wd, backend=backend)
    assert_close(out, ffn_reference(x, wg, wu, wd))
```

Note: `registered_backends("ffn")` only includes a backend after its module is
imported. `xkernels.ops.ffn.__init__` imports the triton/cuda modules on a
best-effort basis, so on a Triton-capable machine `Backend.TRITON` appears here.

- [ ] **Step 2: Run test to verify current state**

Run: `pytest tests/test_ffn.py -v`
Expected: existing 3 tests PASS; parametrized GPU tests are SKIPPED on a CPU-only box, or (on a GPU box before this task's impl) collect with no TRITON entry. This confirms the harness is wired before adding the kernel.

- [ ] **Step 3: Write the Triton kernel + backend**

Create `src/xkernels/ops/ffn/triton/__init__.py`:

```python
"""Triton backends for FFN."""
```

Create `src/xkernels/ops/ffn/triton/ffn_kernel.py`:

```python
"""Triton FFN backend: torch matmuls for the projections, a fused Triton
kernel for the elementwise SwiGLU activation `silu(g) * u`."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..._backends import Backend
from ..._dispatch import register
from .._activation import SwigluAct


@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(g_ptr + offs, mask=mask)
    u = tl.load(u_ptr + offs, mask=mask)
    out = (g * tl.sigmoid(g.to(tl.float32)).to(g.dtype)) * u
    tl.store(out_ptr + offs, out, mask=mask)


def _swiglu_triton(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    g = g.contiguous()
    u = u.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_kernel[grid](g, u, out, n, BLOCK=1024)
    return out


def ffn_triton(x, w_gate, w_up, w_down):
    g = x @ w_gate
    u = x @ w_up
    h = SwigluAct.apply(g, u, _swiglu_triton)
    return h @ w_down


register("ffn", Backend.TRITON)(ffn_triton)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ffn.py -v`
Expected on a CUDA+Triton machine: reference tests PASS, `test_gpu_backend_matches_reference[...-TRITON]` PASS. On a CPU-only machine: GPU tests SKIPPED, reference tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/ffn/triton tests/test_ffn.py
git commit -m "feat: Triton FFN backend with fused SwiGLU activation"
```

---

## Task 8: FFN CUDA/HIP backend (optional compiled extension)

**Files:**
- Create: `src/xkernels/ops/ffn/cuda/__init__.py`, `src/xkernels/ops/ffn/cuda/ffn.cu`

- [ ] **Step 1: Write the CUDA kernel + pybind**

Create `src/xkernels/ops/ffn/cuda/ffn.cu`:

```cpp
// Fused SwiGLU activation: out = silu(g) * u, elementwise.
// Compiles for CUDA (NVIDIA) and, via torch's hipify, for ROCm (AMD).
#include <torch/extension.h>

template <typename scalar_t>
__global__ void swiglu_act_kernel(
    const scalar_t* __restrict__ g,
    const scalar_t* __restrict__ u,
    scalar_t* __restrict__ out,
    const int64_t n) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float gv = static_cast<float>(g[idx]);
    const float uv = static_cast<float>(u[idx]);
    const float silu = gv / (1.0f + __expf(-gv));
    out[idx] = static_cast<scalar_t>(silu * uv);
  }
}

torch::Tensor swiglu_act(torch::Tensor g, torch::Tensor u) {
  TORCH_CHECK(g.is_cuda() && u.is_cuda(), "inputs must be CUDA tensors");
  g = g.contiguous();
  u = u.contiguous();
  auto out = torch::empty_like(g);
  const int64_t n = g.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      g.scalar_type(), "swiglu_act", [&] {
        swiglu_act_kernel<scalar_t><<<blocks, threads>>>(
            g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n);
      });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu_act", &swiglu_act, "Fused SwiGLU activation (CUDA/HIP)");
}
```

- [ ] **Step 2: Write the Python backend that loads the compiled extension**

Create `src/xkernels/ops/ffn/cuda/__init__.py`:

```python
"""CUDA/HIP FFN backend. Registers only if the compiled extension imports.

The extension is built by `setup.py` (one per kernel type) when a CUDA/ROCm
toolkit is present. On NVIDIA it registers as Backend.CUDA; on AMD as
Backend.HIP. If the extension is absent, importing this module raises and the
backend is simply not registered.
"""
from __future__ import annotations

import torch

from ..._backends import Backend, detect_vendor
from ..._dispatch import register
from .._activation import SwigluAct
from . import _cuda  # compiled extension; ImportError if not built


def _swiglu_cuda(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _cuda.swiglu_act(g, u)


def ffn_cuda(x, w_gate, w_up, w_down):
    g = x @ w_gate
    u = x @ w_up
    h = SwigluAct.apply(g, u, _swiglu_cuda)
    return h @ w_down


# Register under whichever vendor this torch build targets (default to CUDA).
_backend = Backend.HIP if detect_vendor() == "amd" else Backend.CUDA
register("ffn", _backend)(ffn_cuda)
```

- [ ] **Step 3: Verify graceful degradation (no toolkit)**

Run: `python -c "import xkernels; from xkernels._dispatch import registered_backends; print([b.name for b in registered_backends('ffn')])"`
Expected: includes `REFERENCE` (and `TRITON`/`CUDA` only where their deps/build exist). No import error — the `try/except` in `ops/ffn/__init__.py` swallows the missing extension.

- [ ] **Step 4: Verify on a GPU box with toolkit (conditional)**

Run (only where a CUDA/ROCm toolkit is installed):
`XKERNELS_FORCE_BUILD=1 pip install -e . && pytest tests/test_ffn.py -v -k CUDA`
Expected: `test_gpu_backend_matches_reference[...-CUDA]` PASS. (Skipped/absent elsewhere — acceptable.)

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/ffn/cuda
git commit -m "feat: optional CUDA/HIP FFN backend (fused SwiGLU)"
```

---

## Task 9: MoE and comm stubs

**Files:**
- Create: `src/xkernels/ops/moe/__init__.py`, `src/xkernels/ops/moe/interface.py`, `src/xkernels/ops/moe/reference.py`
- Create: `src/xkernels/ops/comm/__init__.py`, `src/xkernels/ops/comm/interface.py`, `src/xkernels/ops/comm/reference.py`
- Create: `tests/test_stubs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stubs.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xkernels.ops.moe'`

- [ ] **Step 3: Write the MoE stub**

Create `src/xkernels/ops/moe/__init__.py`:

```python
"""Mixture-of-Experts kernels (stub — reference only for now)."""
from .interface import moe

__all__ = ["moe"]
```

Create `src/xkernels/ops/moe/reference.py`:

```python
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
```

Create `src/xkernels/ops/moe/interface.py`:

```python
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
```

- [ ] **Step 4: Write the comm stub**

Create `src/xkernels/ops/comm/__init__.py`:

```python
"""Communication kernels (stub — reference only for now)."""
from .interface import all_reduce

__all__ = ["all_reduce"]
```

Create `src/xkernels/ops/comm/reference.py`:

```python
"""Pure-torch reference comm ops (test oracle / default backend).

TODO: add real backends (e.g. NCCL/RCCL custom all-reduce, fused reduce-scatter).
The reference models the single-process semantics: a sum over the shard list.
"""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register


def all_reduce_reference(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum-allreduce semantics: every shard becomes the elementwise sum."""
    total = torch.stack(shards, dim=0).sum(dim=0)
    return [total.clone() for _ in shards]


register("comm", Backend.REFERENCE)(all_reduce_reference)
```

Create `src/xkernels/ops/comm/interface.py`:

```python
"""Public `all_reduce` op (stub — dispatches to reference until kernels land)."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def all_reduce(
    shards: list[torch.Tensor],
    *,
    backend: Backend | str = "auto",
) -> list[torch.Tensor]:
    """Sum-allreduce over a list of shards. See `all_reduce_reference`."""
    return dispatch("comm", shards, backend=backend)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_stubs.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/xkernels/ops/moe src/xkernels/ops/comm tests/test_stubs.py
git commit -m "feat: MoE and comm reference stubs with public ops"
```

---

## Task 10: Benchmark script + example

**Files:**
- Create: `benchmarks/README.md`, `benchmarks/bench_ffn.py`, `examples/ffn_usage.py`

- [ ] **Step 1: Write the benchmark script**

Create `benchmarks/bench_ffn.py`:

```python
"""Sweep FFN shapes across available backends; print a markdown table.

Usage: python benchmarks/bench_ffn.py [--dtype float16]
"""
from __future__ import annotations

import argparse

import torch

from xkernels import fused_ffn
from xkernels._dispatch import registered_backends
from xkernels.utils.benchmarking import benchmark

SHAPES = [(2048, 4096, 11008), (4096, 4096, 11008), (8192, 8192, 28672)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backends = registered_backends("ffn")

    print(f"| M | d_model | d_ff | " + " | ".join(b.name for b in backends) + " |")
    print("|---|---|---|" + "|".join(["---"] * len(backends)) + "|")
    for M, d_model, d_ff in SHAPES:
        x = torch.randn(M, d_model, device=device, dtype=dtype)
        wg = torch.randn(d_model, d_ff, device=device, dtype=dtype)
        wu = torch.randn(d_model, d_ff, device=device, dtype=dtype)
        wd = torch.randn(d_ff, d_model, device=device, dtype=dtype)
        times = []
        for b in backends:
            try:
                ms = benchmark(lambda b=b: fused_ffn(x, wg, wu, wd, backend=b))
                times.append(f"{ms:.3f}ms")
            except Exception:
                times.append("n/a")  # backend not runnable on this device
        print(f"| {M} | {d_model} | {d_ff} | " + " | ".join(times) + " |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the benchmark README**

Create `benchmarks/README.md`:

```markdown
# Benchmarks

Each `bench_<type>.py` sweeps representative shapes across every registered
backend and prints a markdown timing table.

```bash
python benchmarks/bench_ffn.py --dtype float16
```

Backends only appear if their deps/build are present on the machine
(`reference` always; `triton`/`cuda` on supported hardware). Timing uses
`xkernels.utils.benchmarking.benchmark` (Triton `do_bench` when available).
```

- [ ] **Step 3: Write the usage example**

Create `examples/ffn_usage.py`:

```python
"""Minimal xkernels usage: a differentiable SwiGLU FFN forward+backward."""
import torch

from xkernels import fused_ffn

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4, 8, 512, device=device, requires_grad=True)
w_gate = torch.randn(512, 1024, device=device, requires_grad=True)
w_up = torch.randn(512, 1024, device=device, requires_grad=True)
w_down = torch.randn(1024, 512, device=device, requires_grad=True)

y = fused_ffn(x, w_gate, w_up, w_down)  # backend="auto"
y.sum().backward()

print("output:", tuple(y.shape), "| x.grad:", tuple(x.grad.shape))
```

- [ ] **Step 4: Verify the example runs**

Run: `python examples/ffn_usage.py`
Expected: prints `output: (4, 8, 512) | x.grad: (4, 8, 512)` (runs on CPU via reference if no GPU).

- [ ] **Step 5: Verify the benchmark runs on CPU**

Run: `python benchmarks/bench_ffn.py --dtype float32`
Expected: a markdown table with at least a `REFERENCE` column and one row per shape (slow on CPU — that's fine; it only needs to run).

- [ ] **Step 6: Commit**

```bash
git add benchmarks examples
git commit -m "feat: FFN benchmark sweep and usage example"
```

---

## Task 11: Docs, pre-commit, CI, README

**Files:**
- Create: `docs/adding-a-kernel.md`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `README.md`

- [ ] **Step 1: Write the extension guide**

Create `docs/adding-a-kernel.md`:

````markdown
# Adding a Kernel

`xkernels` is organized **kernel-type first**; each type owns its backends.

## Add a backend to an existing kernel type

1. Create the backend module under `src/xkernels/ops/<type>/<backend>/`.
2. Implement a function with the same signature as the reference, then register it:

   ```python
   from ..._backends import Backend
   from ..._dispatch import register

   @register("<type>", Backend.TRITON)
   def <type>_triton(...):
       ...
   ```
3. Import it for its side effect in `ops/<type>/__init__.py` (guard optional
   deps with `try/except`).
4. Add a parametrized correctness test comparing it to `<type>/reference.py`
   (skip when the hardware/deps are absent — see `tests/test_ffn.py`).

## Add a new kernel type

1. `mkdir src/xkernels/ops/<type>/`, add `reference.py` (the oracle + default
   backend, registered as `Backend.REFERENCE`), `interface.py` (public op that
   calls `dispatch("<type>", ...)`), and `__init__.py` (re-export the op).
2. Re-export from `src/xkernels/__init__.py` if it should be top-level.
3. Add `tests/test_<type>.py` and `benchmarks/bench_<type>.py`.

## CUDA/HIP backends

Drop `.cu` sources in `ops/<type>/cuda/`. `setup.py` builds one extension per
type automatically (`xkernels.ops.<type>.cuda._cuda`). The same source serves
NVIDIA (CUDA) and AMD (ROCm, via torch's hipify); register as `Backend.CUDA` or
`Backend.HIP` based on `detect_vendor()`.

## torch.compile

The reference backend is pure torch and traces under `torch.compile`. To make a
custom-kernel backend fully capturable, promote it to a
`torch.library.custom_op` with a registered fake/meta impl — a deliberate,
per-kernel step, not done by default.
````

- [ ] **Step 2: Write the pre-commit config**

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 3: Write the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test (CPU/reference path)
        run: pytest -v
```

- [ ] **Step 4: Write the README**

Create `README.md`:

````markdown
# xkernels

Customized compute kernels across hardware vendors (NVIDIA, AMD, …) and kernel
types (FFN, MoE, comm, …), with a uniform PyTorch API, automatic backend
dispatch, and a correctness + benchmark harness.

## Install

```bash
pip install -e ".[dev]"          # pure-Python (reference + triton if present)
XKERNELS_FORCE_BUILD=1 pip install -e .   # also build CUDA/HIP extensions
```

Triton/CUDA backends are optional; the package runs on the pure-torch reference
path anywhere.

## Usage

```python
import torch
from xkernels import fused_ffn

y = fused_ffn(x, w_gate, w_up, w_down)            # backend="auto"
y = fused_ffn(x, w_gate, w_up, w_down, backend="triton")  # force a backend
```

Override globally with `XKERNELS_BACKEND=reference|triton|cuda|hip`.

## Layout

- `src/xkernels/ops/<type>/` — kernels by type; each has `reference.py`,
  `interface.py`, and per-backend subdirs (`triton/`, `cuda/`).
- `src/xkernels/_dispatch.py` — backend registry + selection.
- `tests/`, `benchmarks/`, `examples/` — harness and demos.

See `docs/adding-a-kernel.md` to extend. Design: `docs/superpowers/specs/`.
````

- [ ] **Step 5: Run lint + full test suite**

Run: `ruff check . && pytest -v`
Expected: ruff reports no errors; all non-GPU tests PASS, GPU tests SKIPPED on a CPU box.

- [ ] **Step 6: Commit**

```bash
git add docs/adding-a-kernel.md .pre-commit-config.yaml .github/workflows/ci.yml README.md
git commit -m "docs: extension guide, README, pre-commit, and CI"
```

---

## Final Verification

- [ ] `ruff check .` — clean
- [ ] `pytest -v` — reference/CPU tests pass; GPU tests skip cleanly on a CPU box
- [ ] `python examples/ffn_usage.py` — prints expected shapes
- [ ] `python benchmarks/bench_ffn.py --dtype float32` — prints a table
- [ ] `python -c "import xkernels; print(xkernels.fused_ffn)"` — imports cleanly
- [ ] On a GPU box (if available): `XKERNELS_FORCE_BUILD=1 pip install -e . && pytest -v` — Triton (and CUDA) backend tests pass
