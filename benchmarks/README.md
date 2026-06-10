# Benchmarks

Each `bench_<type>.py` sweeps representative shapes across every registered
backend and prints a markdown timing table.

```bash
python benchmarks/bench_ffn.py --dtype float16
```

Backends only appear if their deps/build are present on the machine
(`reference` always; `triton`/`cuda` on supported hardware). Timing uses
`xkernels.utils.benchmarking.benchmark` (Triton `do_bench` when available).
