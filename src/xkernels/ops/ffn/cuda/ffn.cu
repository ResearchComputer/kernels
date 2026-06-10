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
