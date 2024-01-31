#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
at::Tensor bwd_forget_mult_cuda_forward(at::Tensor x, at::Tensor f, at::Tensor output, bool batch_first);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor bwd_forget_mult_forward(at::Tensor x, at::Tensor f, at::Tensor output, bool batch_first) {
  CHECK_INPUT(x); CHECK_INPUT(f); CHECK_INPUT(output);
  return bwd_forget_mult_cuda_forward(x, f, output, batch_first);
}

std::vector<at::Tensor> bwd_forget_mult_cuda_backward(at::Tensor x, at::Tensor f, at::Tensor output,
                at::Tensor grad_output, bool batch_first);

std::vector<at::Tensor> bwd_forget_mult_backward(at::Tensor x, at::Tensor f, at::Tensor output,
                at::Tensor grad_output, bool batch_first) {
  CHECK_INPUT(x); CHECK_INPUT(f); CHECK_INPUT(output);
  return bwd_forget_mult_cuda_backward(x, f, output, grad_output, batch_first);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bwd_forget_mult_forward, "BwdForgetMult forward (CUDA)");
  m.def("backward", &bwd_forget_mult_backward, "BwdForgetMult backward (CUDA)");
}
