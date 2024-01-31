#include <ATen/ATen.h>
#include <THC/THC.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void forget_mult_cuda_forward_kernel(const scalar_t* __restrict__ x,
                const scalar_t* __restrict__ f, scalar_t* __restrict__ output,
                size_t batch_size, size_t seq_length, size_t n_hidden, bool batch_first) {
  /*
  Note: output is assumed to be one timestep longer than f or x where output[0] = h_{-1}
  This means output array has a size of seq_length+1 on the word dimension
  */
  const int hid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if (hid < n_hidden && bid < batch_size){
    for (int ts = 1; ts < seq_length + 1; ts++) {
      int i           = 0;
      int dst_i       = 0;
      int dst_iminus1 = 0;
      if (batch_first){
        i           = bid * n_hidden * seq_length     + (ts-1) * n_hidden + hid;
        dst_i       = bid * n_hidden * (seq_length+1) + (ts-0) * n_hidden + hid;
        dst_iminus1 = bid * n_hidden * (seq_length+1) + (ts-1) * n_hidden + hid;
      }
      else {
        i           = (ts-1) * n_hidden * batch_size  + bid * n_hidden + hid;
        dst_i       = (ts-0) * n_hidden * batch_size  + bid * n_hidden + hid;
        dst_iminus1 = (ts-1) * n_hidden * batch_size  + bid * n_hidden + hid;
      }
      output[dst_i]   = f[i] * x[i];
      output[dst_i]  += (1 - f[i]) * output[dst_iminus1];
    }
  }
}

template <typename scalar_t>
__global__ void forget_mult_cuda_backward_kernel(const scalar_t* __restrict__ x,
                const scalar_t* __restrict__ f, const scalar_t* __restrict__ output,
                const scalar_t* __restrict__ grad_output, scalar_t* __restrict__ grad_x,
                scalar_t* __restrict__ grad_f, scalar_t* __restrict__ grad_h,
                size_t batch_size, size_t seq_length, size_t n_hidden, bool batch_first) {
  const int hid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y * blockDim.y + threadIdx.y;
  double running_f = 0;
  if(hid < n_hidden && bid < batch_size){
    for (int ts = seq_length; ts >= 0 + 1; ts--) {
      int i           = 0;
      int dst_i       = 0;
      int dst_iminus1 = 0;
      if (batch_first){
        i           = bid * n_hidden * seq_length     + (ts-1) * n_hidden + hid;
        dst_i       = bid * n_hidden * (seq_length+1) + (ts-0) * n_hidden + hid;
        dst_iminus1 = bid * n_hidden * (seq_length+1) + (ts-1) * n_hidden + hid;
      }
      else {
        i           = (ts-1) * n_hidden * batch_size  + bid * n_hidden + hid;
        dst_i       = (ts-0) * n_hidden * batch_size  + bid * n_hidden + hid;
        dst_iminus1 = (ts-1) * n_hidden * batch_size  + bid * n_hidden + hid;
      }
      running_f       += grad_output[i];
      grad_x[i]       = f[i] * running_f;
      grad_f[i]       = (x[i] - output[dst_iminus1]) * running_f;
      // The line below is likely more numerically stable than (1 - f[i]) * running_f;
      running_f       = running_f - f[i] * running_f;
    }
    grad_h[bid * n_hidden + hid] = running_f;
  }
}

at::Tensor forget_mult_cuda_forward(at::Tensor x, at::Tensor f, at::Tensor output, bool batch_first) {
  const auto batch_size = (batch_first) ? x.size(0) : x.size(1);
  const auto seq_length = (batch_first) ? x.size(1) : x.size(0);
  const auto n_hidden   = x.size(2);
  
  const int threads = 1024;
  const dim3 blocks((n_hidden + threads - 1) / threads, batch_size);
  AT_DISPATCH_FLOATING_TYPES(x.type(), "forget_mult_cuda_forward", ([&] {
    forget_mult_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(), f.data<scalar_t>(), output.data<scalar_t>(), batch_size,
        seq_length, n_hidden, batch_first);
  }));

  THCudaCheck(cudaGetLastError());
  return output;
}

std::vector<at::Tensor> forget_mult_cuda_backward(at::Tensor x, at::Tensor f,
                at::Tensor output, at::Tensor grad_output, bool batch_first) {
  const auto batch_size = (batch_first) ? x.size(0) : x.size(1);
  const auto seq_length = (batch_first) ? x.size(1) : x.size(0);
  const auto n_hidden   = x.size(2);

  auto grad_x = at::zeros_like(x);
  auto grad_f = at::zeros_like(x);
  auto grad_h = at::zeros({batch_size, n_hidden}, x.options());
  
  const int threads = 1024;
  const dim3 blocks((n_hidden + threads - 1) / threads, batch_size);
  AT_DISPATCH_FLOATING_TYPES(x.type(), "forget_mult_cuda_forward", ([&] {
    forget_mult_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(), f.data<scalar_t>(), output.data<scalar_t>(), grad_output.data<scalar_t>(),
        grad_x.data<scalar_t>(), grad_f.data<scalar_t>(), grad_h.data<scalar_t>(), batch_size,
        seq_length, n_hidden, batch_first);
  }));

  THCudaCheck(cudaGetLastError());
  return {grad_x, grad_f, grad_h};
}

