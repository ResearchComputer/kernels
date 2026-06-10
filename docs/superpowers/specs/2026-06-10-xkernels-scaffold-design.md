# xkernels — Repository Scaffold Design

**Date:** 2026-06-10
**Status:** Approved
**Import name:** `xkernels`

## Purpose

A repository for storing customized compute kernels, spanning multiple hardware
vendors (NVIDIA, AMD, and more later) and multiple kernel types (FFN, MoE, comm,
and more later). It serves a dual role:

1. **Importable library** — a clean PyTorch-facing package others can `import`,
   with a stable public API that dispatches to the right backend for the device.
2. **Research harness** — first-class correctness tests and benchmarks around
   every kernel, so implementations can be compared and validated.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Authoring styles | Mixed: Triton + CUDA/HIP C++ + DSLs (CUTLASS/TK) | Scaffold must not assume one style; each kernel type hosts multiple backends. |
| Framework | PyTorch only | `torch.Tensor` in/out; custom ops via `torch.library` / `autograd.Function`. |
| Organization axis | **Kernel-type first**, backend as sub-layer | Matches user mental model ("give me a fused FFN"); keeps cross-vendor Triton kernels DRY; adding a type or a vendor is additive. |
| Purpose | Library **and** benchmark/test harness | Covers the full lifecycle. |
| Package name | `xkernels` | Avoids collision with HuggingFace's `kernels` on PyPI. |
| License | MIT | Permissive; standard for shareable kernel libraries. |

## Architecture

Kernel **type** is the top axis. Each type exposes one public entry point that
dispatches to a backend (Triton / CUDA / HIP / reference) chosen from the runtime
device and vendor. A single Triton source covers both NVIDIA and AMD, so it is
written once, not duplicated per vendor.

### Directory layout

```
kernels/                            # repo root (git)
├── pyproject.toml                  # metadata, deps, optional-deps groups, tool config
├── setup.py                        # custom build_ext to compile CUDA/HIP extensions
├── README.md  LICENSE  .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml        # lint + CPU/reference tests
├── docs/
│   ├── adding-a-kernel.md          # how-to-extend guide (key to usability)
│   └── superpowers/specs/          # design docs (this file)
├── src/xkernels/
│   ├── __init__.py                 # re-exports public ops + version
│   ├── _backends.py                # Backend enum + vendor/device detection
│   ├── _dispatch.py                # registry + backend selection
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── ffn/                     # fully worked example
│   │   │   ├── __init__.py          # public: fused_ffn(...)
│   │   │   ├── interface.py         # signature, autograd, dispatch call
│   │   │   ├── reference.py         # pure-torch impl (test oracle)
│   │   │   ├── triton/
│   │   │   │   ├── __init__.py
│   │   │   │   └── ffn_kernel.py
│   │   │   └── cuda/
│   │   │       ├── ffn.cu           # compiled when toolkit present
│   │   │       └── bind.cpp
│   │   ├── moe/                     # stub: __init__ + interface + reference (TODO impls)
│   │   │   ├── __init__.py
│   │   │   ├── interface.py
│   │   │   └── reference.py
│   │   └── comm/                    # stub
│   │       ├── __init__.py
│   │       ├── interface.py
│   │       └── reference.py
│   └── utils/
│       ├── __init__.py
│       ├── benchmarking.py          # do_bench / cuda-event timing helpers
│       └── testing.py               # assert_close + per-dtype tolerance presets
├── benchmarks/
│   ├── README.md
│   └── bench_ffn.py                 # shape × backend sweeps → CSV/markdown
├── tests/
│   ├── conftest.py                  # device + available-backends fixtures
│   └── test_ffn.py
└── examples/
    └── ffn_usage.py
```

CUDA/HIP source lives next to its kernel type (`ops/<type>/cuda/`), not in a
separate top-level `csrc/`, keeping each kernel type cohesive.

**CUDA vs. HIP backends:** by default a single `cuda/*.cu` source serves both
vendors — `torch.utils.cpp_extension` auto-hipifies it under a ROCm install. The
compiled extension registers as `Backend.CUDA` on NVIDIA and `Backend.HIP` on
AMD (the build detects the toolchain). A dedicated `ops/<type>/hip/` directory is
added only when a kernel needs genuinely AMD-specific source that hipify cannot
produce; the FFN example does not need one.

## Components

### Dispatch layer

- **`_backends.py`** — a `Backend` enum (`TRITON`, `CUDA`, `HIP`, `REFERENCE`)
  and detection helpers: NVIDIA vs AMD via `torch.version.cuda` /
  `torch.version.hip`, combined with the input tensor's device.
- **`_dispatch.py`** — a lightweight registry. Each backend impl self-registers
  with `@register("ffn", Backend.TRITON)`. The public op calls
  `dispatch("ffn", ..., backend="auto")`, which resolves in order:
  **explicit arg → env override (`XKERNELS_BACKEND`) → auto** (a per-vendor
  preference order), falling back to `REFERENCE` on CPU or unsupported devices.
- Custom kernels are wrapped in `torch.autograd.Function` so the public op is
  end-to-end differentiable (projection matmuls use `torch.matmul`; the custom
  kernel handles the fused activation, with a torch-computed backward). The
  reference backend is pure torch and `torch.compile`-traceable; promoting a
  custom-kernel backend to a `torch.library.custom_op` for full graph capture is
  a documented extension point in `docs/adding-a-kernel.md`.

Extending: adding a backend = drop a file + `@register` (no edits to dispatch
core). Adding a vendor = add an enum value + a detection rule.

### Test + benchmark harness

- **`tests/`** — pytest. Each kernel type compares every *available* backend
  against its `reference.py`, parametrized over dtypes and shapes. Backends not
  available on the current hardware are **skipped, not failed**, so one suite
  runs unchanged on any machine.
- **`benchmarks/`** — standalone scripts sweeping shapes × backends, timing via
  `utils/benchmarking.py` (Triton `do_bench` when present, CUDA events
  otherwise), emitting CSV/markdown for later plotting.
- **`utils/testing.py`** — shared `assert_close` with fp16/bf16/fp32 tolerance
  presets so tests stay consistent.

### Build & packaging

- `pyproject.toml` with optional-dependency groups: `[dev]` (ruff, pytest,
  pre-commit), `[bench]` (plotting deps), `[triton]`. Lint + format via **ruff**.
- Triton and reference kernels need **no build step** (pure Python). CUDA/HIP
  extensions are **optional**, compiled by `setup.py` via
  `torch.utils.cpp_extension` (auto-hipifies for ROCm). The package imports and
  runs Triton-only when no compiler/toolkit is present — compiled backends simply
  do not register.
- CI runs lint + the CPU/reference path of the test suite (GPU tests gated,
  since hosted runners lack GPUs).

## Scope: scaffolded vs. stubbed

- **FFN** — fully worked: Triton kernel + reference + test + benchmark. This is
  the copy-able template.
- **MoE, comm** — directory + `interface.py` + `reference.py` + `TODO`,
  demonstrating the shape without writing real kernels yet.
- **`docs/adding-a-kernel.md`** — walks through extending both axes (new backend
  for an existing type; new kernel type).

## Non-goals (YAGNI)

- No JAX / framework-agnostic core yet — PyTorch only, kept additive.
- No cookiecutter/template generator — the worked FFN example + doc suffice.
- No real MoE/comm kernel implementations in this scaffold — stubs only.
- No GPU CI — gated; correctness is validated locally on real hardware.

## Success criteria

- `pip install -e .` works with Triton-only (no CUDA toolkit required).
- `import xkernels; xkernels.fused_ffn(...)` runs and dispatches correctly on
  CPU (reference) and GPU (triton).
- `pytest` passes on a machine with no GPU (reference path) and on a GPU machine
  (reference + triton, CUDA if built).
- Adding a new kernel type or backend follows `docs/adding-a-kernel.md` without
  touching dispatch internals.
