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
