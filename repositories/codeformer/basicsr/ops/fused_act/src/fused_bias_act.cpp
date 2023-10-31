// from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act.cpp
#include <torch/extension.h>


torch::Tensor fused_bias_act_op(const torch::Tensor& input,
                                const torch::Tensor& bias,
                                const torch::Tensor& refer,
                                int act, int grad, float alpha, float scale);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_bias_act(const torch::Tensor& input,
                             const torch::Tensor& bias,
                             const torch::Tensor& refer,
                             int act, int grad, float alpha, float scale) {
    CHECK_CUDA(input);
    CHECK_CUDA(bias);

    return fused_bias_act_op(input, bias, refer, act, grad, alpha, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_act", &fused_bias_act, "fused bias act (CUDA)");
}
