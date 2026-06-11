# Fused top-k combine epilogue for INT4 W4A16 MoE GEMM (issue #20) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `fused_combine` mode that atomic-accumulates the routing-weighted INT4-MoE GEMM result directly into an `[M, hidden]` fp32 output in the kernel epilogue, eliminating the separate `moe_sum_reduce` launch and the `[M*top_k, hidden]` intermediate.

**Architecture:** A single `COMBINE: tl.constexpr` flag on the existing `_fused_moe_int4_kernel`. When set, the epilogue computes the output row `offs_token // top_k` and `tl.atomic_add`s the (fp32) weighted accumulator into an `[M, N]` fp32 buffer instead of `tl.store`-ing the token-indexed `[M*top_k, N]` buffer. `compute_type=float32` makes the existing cast match the buffer. Default `COMBINE=False` keeps every existing caller byte-for-byte unchanged; `moe_sum_reduce` stays for un-reduced consumers. `tl.atomic_add` is verified to work under `TRITON_INTERPRET=1`, so correctness is testable on CPU.

**Tech Stack:** Python 3.11, Triton 3.7 (ROCm fork on device), PyTorch, SLURM/enroot on CSCS beverin (MI300A, gfx942).

**Local commands:** Python via `.venv/bin/python`; tests via `TRITON_INTERPRET=1 .venv/bin/python -m pytest ...`; lint via `.venv/bin/ruff check .`.

---

## File structure

- **Modify** `src/xkernels/ops/moe/triton/moe_int4_kernel.py` — `COMBINE` constexpr + epilogue/filtered-branch; `int4_w4a16_moe_gemm(combine=False)`; `_moe_int4_w4a16_triton(fused_combine=False)`.
- **Modify** `src/xkernels/ops/moe/interface.py` — `fused_moe_int4_w4a16(fused_combine=False)` passthrough.
- **Modify** `src/xkernels/ops/moe/reference.py` — `moe_w4a16_ref` accepts (and ignores) `fused_combine` so dispatch can forward it.
- **Modify** `tests/test_moe_int4_w4a16.py` — fused-combine correctness test.
- **Create** `benchmarks/bench_moe_combine.py` + `slurm/bench_moe_combine_beverin.sbatch` — on-device fused-vs-(GEMM+reduce) comparison.

---

### Task 1: `COMBINE` kernel path + `fused_combine` plumbing

**Files:**
- Modify: `src/xkernels/ops/moe/triton/moe_int4_kernel.py`
- Modify: `src/xkernels/ops/moe/interface.py`
- Modify: `src/xkernels/ops/moe/reference.py`
- Test: `tests/test_moe_int4_w4a16.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_int4_w4a16.py`:

```python
@pytest.mark.parametrize("mul_routed", [False, True])
def test_fused_combine_matches_reference(mul_routed):
    """fused_combine=True (atomic top-k combine in the epilogue) == GEMM+reduce."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered (triton not installed)")
    dev = _device()
    group_size = 32
    _pin_single_config()
    M, E, N, K, top_k = 8, 8, 256, 512, 4
    packed, scale, A, topk_ids, topk_w = _inputs(M, E, N, K, top_k, dev, group_size)
    got = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=mul_routed,
        backend=Backend.TRITON, fused_combine=True,
    )
    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed)
    assert got.shape == (M, N)  # [M, N], no [M*top_k, N] intermediate exposed
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)
```

- [ ] **Step 2: Run to verify it fails**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest "tests/test_moe_int4_w4a16.py::test_fused_combine_matches_reference" -q`
Expected: FAIL — `fused_moe_int4_w4a16()` got an unexpected keyword argument `fused_combine`.

- [ ] **Step 3: Add `COMBINE` to the kernel signature**

In `src/xkernels/ops/moe/triton/moe_int4_kernel.py`, in `_fused_moe_int4_kernel`'s signature, change:

```python
    FILTER_EXPERT: tl.constexpr,
    # AMD/CDNA3 lowering knobs. Declared as (unused) constexpr so the same
```

to:

```python
    FILTER_EXPERT: tl.constexpr,
    COMBINE: tl.constexpr = False,
    # AMD/CDNA3 lowering knobs. Declared as (unused) constexpr so the same
```

- [ ] **Step 4: Branch the filtered-block early-return on `COMBINE`**

Replace:

```python
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if FILTER_EXPERT and off_experts == -1:
        # Filtered (EP) block: write zeros and exit. At EP=8 most blocks for a
        # given rank are *not* filtered, but routed tokens whose expert lives on
        # another rank still produce a -1 block that must zero its output slot.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return
```

with:

```python
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if FILTER_EXPERT and off_experts == -1:
        # Filtered (EP) block. In the default path write zeros to the token-indexed
        # slot; in COMBINE mode the [M, N] buffer is pre-zeroed, so just exit.
        if not COMBINE:
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(
                c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask
            )
        return
```

- [ ] **Step 5: Branch the epilogue store on `COMBINE`**

Replace:

```python
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

with:

```python
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if COMBINE:
        # Fused weighted top-k combine: atomic-accumulate each expert's result into
        # the token's row (offs_token // top_k) of the [M, N] fp32 output. Padding
        # slots are masked out, so they never touch a valid row.
        out_rows = offs_token // top_k
        c_ptrs = c_ptr + stride_cm * out_rows[:, None] + stride_cn * offs_cn[None, :]
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
```

- [ ] **Step 6: Thread `combine` through `int4_w4a16_moe_gemm`**

Change the signature (add `combine`):

```python
    filter_expert: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
```

to:

```python
    filter_expert: bool = True,
    config: dict | None = None,
    combine: bool = False,
) -> torch.Tensor:
```

Add an fp32 guard right after the existing `assert b_scale.shape == ...` line:

```python
    assert b_scale.shape == (E, N, K // group_size)
    if combine:
        assert c.dtype == torch.float32, "combine mode requires an fp32 output buffer"
```

Add `COMBINE=combine` to the `common_kw` dict:

```python
    common_kw = dict(
        group_k=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=compute_type,
        FILTER_EXPERT=filter_expert,
        COMBINE=combine,
    )
```

(Both launch paths already spread `**common_kw`, so `COMBINE` reaches the tuned-direct launch and the autotuned fallback.)

- [ ] **Step 7: Add `fused_combine` to the registered wrapper**

Replace the `_moe_int4_w4a16_triton` body's buffer-alloc + launch (from `c = torch.zeros(...)` through `return c.view(...)`) — i.e. change the signature and the tail. Change the signature:

```python
def _moe_int4_w4a16_triton(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    group_size: int = 32,
    mul_routed_weight: bool = True,
) -> torch.Tensor:
```

to:

```python
def _moe_int4_w4a16_triton(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    group_size: int = 32,
    mul_routed_weight: bool = True,
    fused_combine: bool = False,
) -> torch.Tensor:
```

Then replace:

```python
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    c = torch.zeros((M * top_k, N), dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float32
    int4_w4a16_moe_gemm(
        A,
        packed,
        scale,
        c,
        topk_w.reshape(-1).float(),
        sorted_ids,
        expert_ids,
        num_post,
        top_k=top_k,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        compute_type=compute_type,
        filter_expert=False,
        config=config,
    )
    return c.view(M, top_k, N).sum(dim=1)
```

with:

```python
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    if fused_combine:
        # Fused weighted top-k combine: the kernel atomic-accumulates into a single
        # [M, N] fp32 buffer, so there is no [M*top_k, N] scratch and no separate
        # moe_sum_reduce. fp32 accumulate -> cast to the activation dtype.
        out = torch.zeros((M, N), dtype=torch.float32, device=A.device)
        int4_w4a16_moe_gemm(
            A,
            packed,
            scale,
            out,
            topk_w.reshape(-1).float(),
            sorted_ids,
            expert_ids,
            num_post,
            top_k=top_k,
            group_size=group_size,
            mul_routed_weight=mul_routed_weight,
            compute_type=tl.float32,
            filter_expert=False,
            config=config,
            combine=True,
        )
        return out.to(A.dtype)
    c = torch.zeros((M * top_k, N), dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float32
    int4_w4a16_moe_gemm(
        A,
        packed,
        scale,
        c,
        topk_w.reshape(-1).float(),
        sorted_ids,
        expert_ids,
        num_post,
        top_k=top_k,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        compute_type=compute_type,
        filter_expert=False,
        config=config,
    )
    return c.view(M, top_k, N).sum(dim=1)
```

- [ ] **Step 8: Thread `fused_combine` through the public op + reference**

In `src/xkernels/ops/moe/interface.py`, add `fused_combine` to `fused_moe_int4_w4a16` (after `mul_routed_weight`):

```python
    *,
    group_size: int = 32,
    mul_routed_weight: bool = True,
    fused_combine: bool = False,
    backend: Backend | str = "auto",
) -> torch.Tensor:
```

and add it to the dispatch call:

```python
    return dispatch(
        "moe_int4_w4a16",
        A,
        packed,
        scale,
        topk_ids,
        topk_w,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        fused_combine=fused_combine,
        backend=backend,
    )
```

Add to its docstring (after the `mul_routed_weight:` line):

```python
        fused_combine: fuse the weighted top-k combine into the GEMM epilogue
            (Triton backend) — returns ``[M, N]`` directly with no separate reduce.
```

In `src/xkernels/ops/moe/reference.py`, add `fused_combine` to `moe_w4a16_ref` (it already returns the combined `[M, N]`, so the flag is accepted and ignored):

```python
    group_size: int = 32,
    mul_routed_weight: bool = True,
    fused_combine: bool = False,
) -> torch.Tensor:
```

and note it in the docstring (after `mul_routed_weight:`):

```python
        fused_combine: accepted for API parity with the Triton backend; the
            reference already returns the combined ``[M, N]`` result, so it is a
            no-op here.
```

- [ ] **Step 9: Run the fused-combine test (expect pass)**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest "tests/test_moe_int4_w4a16.py::test_fused_combine_matches_reference" -q`
Expected: PASS (2). (`tl.atomic_add` is supported under the interpreter — verified.)

- [ ] **Step 10: Run the full INT4 MoE suite (no regression)**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_w4a16.py tests/test_moe_int4_tuned_config.py -q`
Expected: all PASS (the default `COMBINE=False` path is unchanged).

- [ ] **Step 11: Lint + commit**

```bash
.venv/bin/ruff check src/xkernels/ops/moe/triton/moe_int4_kernel.py src/xkernels/ops/moe/interface.py src/xkernels/ops/moe/reference.py tests/test_moe_int4_w4a16.py
git add src/xkernels/ops/moe/triton/moe_int4_kernel.py src/xkernels/ops/moe/interface.py src/xkernels/ops/moe/reference.py tests/test_moe_int4_w4a16.py
git commit -m "feat(moe): fused weighted top-k combine epilogue for INT4 MoE GEMM (issue #20)"
```

---

### Task 2: On-device fused-vs-reduce benchmark + SLURM job

**Files:**
- Create: `benchmarks/bench_moe_combine.py`
- Create: `slurm/bench_moe_combine_beverin.sbatch`

- [ ] **Step 1: Write the benchmark**

Create `benchmarks/bench_moe_combine.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Compare the fused top-k combine epilogue vs the unfused (GEMM + moe_sum_reduce)
path for the INT4 W4A16 MoE down GEMM (issue #20).

Both produce [M, hidden]; the fused path is one kernel and skips the
[M*top_k, hidden] intermediate. Reports per-path latency across decode M and the
kernel-count drop (2 -> 1). Run on gfx942 (slurm/bench_moe_combine_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels.ops.moe import fused_moe_int4_w4a16, make_w4a16_weights

# Kimi-K2.6 per-rank down GEMM (the combine target): N = hidden, K = moe_inter.
KIMI = dict(E=48, N=7168, K=2048, TOP_K=8, GS=32)
DECODE_M = [1, 2, 4, 8, 16]


def _inputs(M, dev):
    packed, scale, _ = make_w4a16_weights(KIMI["E"], KIMI["N"], KIMI["K"], KIMI["GS"], device=dev, seed=1)
    A = (torch.randn(M, KIMI["K"], device=dev) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(KIMI["E"], device=dev)[: KIMI["TOP_K"]] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, KIMI["TOP_K"], device=dev, dtype=torch.float32)
    return packed, scale, A, topk_ids, topk_w


def main():
    if not torch.cuda.is_available():
        print("No GPU; this benchmark needs gfx942 (or any CUDA/ROCm GPU).")
        return
    import triton

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"down GEMM: E={KIMI['E']} N={KIMI['N']} K={KIMI['K']} top_k={KIMI['TOP_K']}")
    print(f"{'M':>4} {'unfused_ms':>11} {'fused_ms':>9} {'speedup':>8}  (2 kernels -> 1)")
    for M in DECODE_M:
        packed, scale, A, topk_ids, topk_w = _inputs(M, "cuda")

        def unfused(A=A, packed=packed, scale=scale, topk_ids=topk_ids, topk_w=topk_w):
            return fused_moe_int4_w4a16(
                A, packed, scale, topk_ids, topk_w, group_size=KIMI["GS"],
                mul_routed_weight=True, backend="triton", fused_combine=False,
            )

        def fused(A=A, packed=packed, scale=scale, topk_ids=topk_ids, topk_w=topk_w):
            return fused_moe_int4_w4a16(
                A, packed, scale, topk_ids, topk_w, group_size=KIMI["GS"],
                mul_routed_weight=True, backend="triton", fused_combine=True,
            )

        # Correctness guard: the two paths must agree before we trust the timing.
        d = (unfused().float() - fused().float()).abs().max().item()
        u = triton.testing.do_bench(unfused, warmup=10, rep=50)
        f = triton.testing.do_bench(fused, warmup=10, rep=50)
        print(f"{M:4d} {u:11.4f} {f:9.4f} {u / f:8.2f}  (max|err|={d:.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the no-GPU guard + lint**

Run: `.venv/bin/python benchmarks/bench_moe_combine.py`
Expected: prints `No GPU; this benchmark needs gfx942 ...` and exits 0.
Run: `.venv/bin/ruff check benchmarks/bench_moe_combine.py`
Expected: no errors.

- [ ] **Step 3: Write the SLURM job**

Create `slurm/bench_moe_combine_beverin.sbatch`:

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Fused top-k combine vs (GEMM + moe_sum_reduce) for the INT4 MoE down GEMM on
# beverin (gfx942 / MI300A) — issue #20.
#
#   sbatch slurm/bench_moe_combine_beverin.sbatch
#
#SBATCH --job-name=xk-moe-combine
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=moe-combine-%j.out
#SBATCH --error=moe-combine-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  echo "=== fused-combine bench ==="
  python -u "'"$REPO"'/benchmarks/bench_moe_combine.py"
  echo "=== GPU correctness (fused_combine) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_int4_w4a16.py" -q
'
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_moe_combine.py slurm/bench_moe_combine_beverin.sbatch
git commit -m "bench(moe): fused-combine vs GEMM+reduce for INT4 MoE down GEMM (issue #20)"
```

---

### Task 3: Run on beverin, verify, PR

This task runs on the cluster; no TDD.

- [ ] **Step 1: Sync the branch to beverin scratch**

```bash
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.ruff_cache' --exclude '.pytest_cache' \
  ./ beverin:/capstor/scratch/cscs/xyao/kernels/
```
Expected: completes without error. (Renew the CSCS SSH cert at https://sshservice.cscs.ch/ if expired.)

- [ ] **Step 2: Submit the bench/correctness job**

```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch slurm/bench_moe_combine_beverin.sbatch'
```
Expected: `Submitted batch job <JOBID>`.

- [ ] **Step 3: Wait and read the log**

```bash
ssh beverin 'squeue -j <JOBID> -h -o %T; tail -n 60 /capstor/scratch/cscs/xyao/kernels/moe-combine-<JOBID>.out'
```
Expected: the per-M table with `max|err|` ≤ ~0.03 (bf16), the fused-vs-unfused latency + speedup, and the GPU pytest run passing (incl. `test_fused_combine_matches_reference`). If `max|err|` is large or the fused path is materially slower, stop and investigate before claiming success.

- [ ] **Step 4: Run the full local interpreter suite once more**

```bash
TRITON_INTERPRET=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```
Expected: all pass, lint clean.

- [ ] **Step 5: Push, open PR, report on the issue**

```bash
git push -u origin issue-20-fused-combine-epilogue
gh pr create --repo ResearchComputer/kernels --base main \
  --title "feat(moe): fused weighted top-k combine epilogue for INT4 MoE GEMM (issue #20)" \
  --body "<summary: opt-in COMBINE atomic-accumulate; eliminates moe_sum_reduce + [M*top_k,N] intermediate; on-device fused-vs-reduce numbers; references #20>"
```
Then comment on issue #20 with the fused-vs-reduce latency table and the kernel-count/traffic reduction. (Squash-merge per repo convention once reviewed.)

---

## Self-review

- **Spec coverage:** `COMBINE` epilogue atomic-accumulate into `[M,N]` fp32 (Task 1 Steps 3–6) ✓; output row `offs_token // top_k` (Step 5) ✓; fp32 buffer + `compute_type=float32` + cast (Steps 6–7) ✓; filtered-block no-op in combine (Step 4) ✓; `fused_combine` public flag + reference parity (Step 8) ✓; default off / `moe_sum_reduce` kept (defaults `False`, untouched op) ✓; interpreter + GPU correctness (Steps 9–10, Task 3 Step 3) ✓; on-device fused-vs-reduce bench (Task 2 + Task 3) ✓; result on #20 (Task 3 Step 5) ✓. Atomic-under-interpreter risk resolved (verified working) ✓.
- **Placeholder scan:** none — concrete code/commands throughout. `<JOBID>` / PR body are runtime values.
- **Type/name consistency:** `combine` (launcher) ↔ `COMBINE` (kernel constexpr) ↔ `fused_combine` (wrapper + public + reference) used consistently; `COMBINE=combine` in `common_kw`; the combine buffer is fp32 in the wrapper alloc, the `assert c.dtype == torch.float32` guard, and `compute_type=tl.float32`. Output row `offs_token // top_k` matches the A-gather `offs_token // top_k` already in the kernel body.
