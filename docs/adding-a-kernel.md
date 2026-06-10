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
