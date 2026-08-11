// Fused inference-only GroupNorm + SiLU for contiguous float16 MPS tensors.
#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace {

struct FusedGroupNormParams {
  uint32_t batch;
  uint32_t channels;
  uint32_t spatial;
  uint32_t groups;
  float epsilon;
};

static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static inline size_t getMTLBufferOffset(const at::Tensor& tensor) {
  return tensor.storage_offset() * tensor.element_size();
}

static id<MTLComputePipelineState> getFusedGroupNormSiLUPipeline() {
  static id<MTLComputePipelineState> pipeline = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSString* source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct FusedGroupNormParams {
  uint batch;
  uint channels;
  uint spatial;
  uint groups;
  float epsilon;
};

kernel void fused_group_norm_silu_half(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant FusedGroupNormParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float partial_sum[256];
  threadgroup float partial_square_sum[256];

  const uint channels_per_group = params.channels / params.groups;
  const uint group_elements = channels_per_group * params.spatial;
  const uint batch_index = group_index / params.groups;
  const uint channel_group = group_index - batch_index * params.groups;
  const uint base =
      (batch_index * params.channels + channel_group * channels_per_group) * params.spatial;

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (uint index = tid; index < group_elements; index += threads) {
    const float value = float(input[base + index]);
    sum += value;
    square_sum += value * value;
  }
  partial_sum[tid] = sum;
  partial_square_sum[tid] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sum[tid] += partial_sum[tid + stride];
      partial_square_sum[tid] += partial_square_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float mean = partial_sum[0] / float(group_elements);
  const float variance =
      max(partial_square_sum[0] / float(group_elements) - mean * mean, 0.0f);
  const float inverse_stddev = rsqrt(variance + params.epsilon);

  for (uint index = tid; index < group_elements; index += threads) {
    const uint local_channel = index / params.spatial;
    const uint channel = channel_group * channels_per_group + local_channel;
    float value = (float(input[base + index]) - mean) * inverse_stddev;
    value = value * float(weight[channel]) + float(bias[channel]);
    value = value / (1.0f + exp(-value));
    output[base + index] = half(value);
  }
}
)METAL";

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(
        library != nil,
        "Failed to compile fused GroupNorm+SiLU Metal library: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
    id<MTLFunction> function = [library newFunctionWithName:@"fused_group_norm_silu_half"];
    TORCH_CHECK(function != nil, "Fused GroupNorm+SiLU Metal function was not found");
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(
        pipeline != nil,
        "Failed to create fused GroupNorm+SiLU pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
  });
  return pipeline;
}

torch::Tensor fused_group_norm_silu_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t groups,
    double epsilon) {
  TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(
      weight.device().is_mps() && bias.device().is_mps(),
      "weight and bias must be MPS tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(
      weight.scalar_type() == at::kHalf && bias.scalar_type() == at::kHalf,
      "weight and bias must be float16");
  TORCH_CHECK(input.dim() == 4, "input must be NCHW");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(
      weight.is_contiguous() && bias.is_contiguous(),
      "weight and bias must be contiguous");
  TORCH_CHECK(
      groups > 0 && input.size(1) % groups == 0,
      "channels must be divisible by groups");
  TORCH_CHECK(
      weight.numel() == input.size(1) && bias.numel() == input.size(1),
      "weight and bias must match the channel count");

  auto output = torch::empty_like(input);
  FusedGroupNormParams params = {
      static_cast<uint32_t>(input.size(0)),
      static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(input.size(2) * input.size(3)),
      static_cast<uint32_t>(groups),
      static_cast<float>(epsilon),
  };
  auto pipeline = getFusedGroupNormSiLUPipeline();

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        at::mps::getCurrentMPSStream()->endKernelCoalescing();
        id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:getMTLBufferStorage(input)
                     offset:getMTLBufferOffset(input)
                    atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(weight)
                     offset:getMTLBufferOffset(weight)
                    atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(bias)
                     offset:getMTLBufferOffset(bias)
                    atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output)
                     offset:getMTLBufferOffset(output)
                    atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(params.batch * params.groups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
      }
    });
  }
  return output;
}

} // namespace

void register_fused_ops(pybind11::module_& module) {
  module.def(
      "fused_group_norm_silu_forward",
      &fused_group_norm_silu_forward,
      "Fused Metal GroupNorm and SiLU forward pass",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("bias"),
      pybind11::arg("groups"),
      pybind11::arg("epsilon"));
}
