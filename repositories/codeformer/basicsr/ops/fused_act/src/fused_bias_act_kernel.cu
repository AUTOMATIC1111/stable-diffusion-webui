// from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act_kernel.cu
// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
static __global__ void fused_bias_act_kernel(scalar_t* out, const scalar_t* p_x, const scalar_t* p_b, const scalar_t* p_ref,
    int act, int grad, scalar_t alpha, scalar_t scale, int loop_x, int size_x, int step_b, int size_b, int use_bias, int use_ref) {
    int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

    scalar_t zero = 0.0;

    for (int loop_idx = 0; loop_idx < loop_x && xi < size_x; loop_idx++, xi += blockDim.x) {
        scalar_t x = p_x[xi];

        if (use_bias) {
            x += p_b[(xi / step_b) % size_b];
        }

        scalar_t ref = use_ref ? p_ref[xi] : zero;

        scalar_t y;

        switch (act * 10 + grad) {
            default:
            case 10: y = x; break;
            case 11: y = x; break;
            case 12: y = 0.0; break;

            case 30: y = (x > 0.0) ? x : x * alpha; break;
            case 31: y = (ref > 0.0) ? x : x * alpha; break;
            case 32: y = 0.0; break;
        }

        out[xi] = y * scale;
    }
}


torch::Tensor fused_bias_act_op(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int act, int grad, float alpha, float scale) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    auto x = input.contiguous();
    auto b = bias.contiguous();
    auto ref = refer.contiguous();

    int use_bias = b.numel() ? 1 : 0;
    int use_ref = ref.numel() ? 1 : 0;

    int size_x = x.numel();
    int size_b = b.numel();
    int step_b = 1;

    for (int i = 1 + 1; i < x.dim(); i++) {
        step_b *= x.size(i);
    }

    int loop_x = 4;
    int block_size = 4 * 32;
    int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

    auto y = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_bias_act_kernel", [&] {
        fused_bias_act_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
            y.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            ref.data_ptr<scalar_t>(),
            act,
            grad,
            alpha,
            scale,
            loop_x,
            size_x,
            step_b,
            size_b,
            use_bias,
            use_ref
        );
    });

    return y;
}
