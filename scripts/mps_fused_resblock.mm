#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

namespace {

struct ResBlockParams {
  uint32_t batch;
  uint32_t channels;
  uint32_t height;
  uint32_t width;
  uint32_t groups;
  float epsilon;
};

struct ResBlockPipelines {
  id<MTLComputePipelineState> stats;
  id<MTLComputePipelineState> block;
};

static inline id<MTLBuffer> buffer_storage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static inline size_t buffer_offset(const at::Tensor& tensor) {
  return tensor.storage_offset() * tensor.element_size();
}

static ResBlockPipelines get_pipelines() {
  static ResBlockPipelines pipelines;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSString* source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ResBlockParams {
  uint batch;
  uint channels;
  uint height;
  uint width;
  uint groups;
  float epsilon;
};

kernel void resblock_stats(
    device const half* input [[buffer(0)]],
    device const half* embedding [[buffer(1)]],
    device float* stats [[buffer(2)]],
    constant ResBlockParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[256];
  threadgroup float squares[256];
  const uint spatial = params.height * params.width;
  const uint channels_per_group = params.channels / params.groups;
  const uint elements = channels_per_group * spatial;
  const uint batch_index = group_index / params.groups;
  const uint group = group_index % params.groups;
  const uint base = (batch_index * params.channels + group * channels_per_group) * spatial;
  float sum = 0.0f;
  float square = 0.0f;
  for (uint index = tid; index < elements; index += threads) {
    const uint channel = group * channels_per_group + index / spatial;
    const half value = half(input[base + index] + embedding[batch_index * params.channels + channel]);
    sum += float(value);
    square += float(value) * float(value);
  }
  sums[tid] = sum;
  squares[tid] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
      squares[tid] += squares[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = sums[0] / float(elements);
  const float variance = max(squares[0] / float(elements) - mean * mean, 0.0f);
  stats[group_index * 2] = mean;
  stats[group_index * 2 + 1] = rsqrt(variance + params.epsilon);
}

kernel void resblock_conv(
    device const half* input [[buffer(0)]],
    device const half* embedding [[buffer(1)]],
    device const float* stats [[buffer(2)]],
    device const half* norm_weight [[buffer(3)]],
    device const half* norm_bias [[buffer(4)]],
    device const half* conv_weight [[buffer(5)]],
    device const half* conv_bias [[buffer(6)]],
    device const half* residual [[buffer(7)]],
    device half* output [[buffer(8)]],
    constant ResBlockParams& params [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  const uint spatial = params.height * params.width;
  const uint count = params.batch * params.channels * spatial;
  if (index >= count) return;
  const uint pixel = index % spatial;
  const uint x = pixel % params.width;
  const uint y = pixel / params.width;
  const uint output_channel = (index / spatial) % params.channels;
  const uint batch_index = index / (params.channels * spatial);
  const uint channels_per_group = params.channels / params.groups;
  float result = float(conv_bias[output_channel]);
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      const int source_y = int(y) + ky;
      const int source_x = int(x) + kx;
      if (source_y < 0 || source_y >= int(params.height) || source_x < 0 || source_x >= int(params.width)) continue;
      const uint kernel_index = uint((ky + 1) * 3 + kx + 1);
      for (uint input_channel = 0; input_channel < params.channels; ++input_channel) {
        const uint source_index = (batch_index * params.channels + input_channel) * spatial + uint(source_y) * params.width + uint(source_x);
        const uint group = input_channel / channels_per_group;
        const float mean = stats[(batch_index * params.groups + group) * 2];
        const float inverse = stats[(batch_index * params.groups + group) * 2 + 1];
        const half combined = half(input[source_index] + embedding[batch_index * params.channels + input_channel]);
        float value = (float(combined) - mean) * inverse;
        value = value * float(norm_weight[input_channel]) + float(norm_bias[input_channel]);
        value = value / (1.0f + exp(-value));
        const uint weight_index = ((output_channel * params.channels + input_channel) * 9) + kernel_index;
        result += value * float(conv_weight[weight_index]);
      }
    }
  }
  output[index] = half(result + float(residual[index]));
}
)METAL";
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(library != nil, "Failed to compile ResBlock Metal library: ", error ? [[error localizedDescription] UTF8String] : "unknown error");
    id<MTLFunction> stats_function = [library newFunctionWithName:@"resblock_stats"];
    id<MTLFunction> block_function = [library newFunctionWithName:@"resblock_conv"];
    TORCH_CHECK(stats_function != nil && block_function != nil, "ResBlock Metal functions were not found");
    pipelines.stats = [device newComputePipelineStateWithFunction:stats_function error:&error];
    pipelines.block = [device newComputePipelineStateWithFunction:block_function error:&error];
    TORCH_CHECK(pipelines.stats != nil && pipelines.block != nil, "Failed to create ResBlock Metal pipelines");
  });
  return pipelines;
}

} // namespace

torch::Tensor fused_resblock_forward(
    const torch::Tensor& input,
    const torch::Tensor& embedding,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& residual,
    int64_t groups,
    double epsilon) {
  TORCH_CHECK(input.device().is_mps() && embedding.device().is_mps() && residual.device().is_mps(), "activations must be MPS tensors");
  TORCH_CHECK(norm_weight.device().is_mps() && norm_bias.device().is_mps() && conv_weight.device().is_mps() && conv_bias.device().is_mps(), "weights must be MPS tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf && embedding.scalar_type() == at::kHalf && residual.scalar_type() == at::kHalf, "activations must be float16");
  TORCH_CHECK(norm_weight.scalar_type() == at::kHalf && norm_bias.scalar_type() == at::kHalf && conv_weight.scalar_type() == at::kHalf && conv_bias.scalar_type() == at::kHalf, "weights must be float16");
  TORCH_CHECK(input.dim() == 4 && embedding.dim() == 2 && residual.sizes() == input.sizes(), "invalid activation shapes");
  TORCH_CHECK(input.is_contiguous() && embedding.is_contiguous() && residual.is_contiguous(), "activations must be contiguous");
  TORCH_CHECK(norm_weight.is_contiguous() && norm_bias.is_contiguous() && conv_weight.is_contiguous() && conv_bias.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(embedding.size(0) == input.size(0) && embedding.size(1) == input.size(1), "embedding must be [batch, channels]");
  TORCH_CHECK(norm_weight.numel() == input.size(1) && norm_bias.numel() == input.size(1), "normalization weights must match channels");
  TORCH_CHECK(conv_weight.dim() == 4 && conv_weight.size(0) == input.size(1) && conv_weight.size(1) == input.size(1) && conv_weight.size(2) == 3 && conv_weight.size(3) == 3 && conv_bias.numel() == input.size(1), "convolution weights must be [channels, channels, 3, 3]");
  TORCH_CHECK(groups > 0 && input.size(1) % groups == 0, "channels must be divisible by groups");

  auto output = torch::empty_like(input);
  auto stats = torch::empty({input.size(0), groups, 2}, input.options().dtype(torch::kFloat));
  ResBlockParams params = {
      static_cast<uint32_t>(input.size(0)), static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(input.size(2)), static_cast<uint32_t>(input.size(3)),
      static_cast<uint32_t>(groups), static_cast<float>(epsilon),
  };
  const auto pipelines = get_pipelines();
  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        at::mps::getCurrentMPSStream()->endKernelCoalescing();
        id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> stats_encoder = [command_buffer computeCommandEncoder];
        [stats_encoder setComputePipelineState:pipelines.stats];
        [stats_encoder setBuffer:buffer_storage(input) offset:buffer_offset(input) atIndex:0];
        [stats_encoder setBuffer:buffer_storage(embedding) offset:buffer_offset(embedding) atIndex:1];
        [stats_encoder setBuffer:buffer_storage(stats) offset:buffer_offset(stats) atIndex:2];
        [stats_encoder setBytes:&params length:sizeof(params) atIndex:3];
        [stats_encoder dispatchThreadgroups:MTLSizeMake(params.batch * params.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [stats_encoder endEncoding];
        id<MTLComputeCommandEncoder> block_encoder = [command_buffer computeCommandEncoder];
        [block_encoder setComputePipelineState:pipelines.block];
        [block_encoder setBuffer:buffer_storage(input) offset:buffer_offset(input) atIndex:0];
        [block_encoder setBuffer:buffer_storage(embedding) offset:buffer_offset(embedding) atIndex:1];
        [block_encoder setBuffer:buffer_storage(stats) offset:buffer_offset(stats) atIndex:2];
        [block_encoder setBuffer:buffer_storage(norm_weight) offset:buffer_offset(norm_weight) atIndex:3];
        [block_encoder setBuffer:buffer_storage(norm_bias) offset:buffer_offset(norm_bias) atIndex:4];
        [block_encoder setBuffer:buffer_storage(conv_weight) offset:buffer_offset(conv_weight) atIndex:5];
        [block_encoder setBuffer:buffer_storage(conv_bias) offset:buffer_offset(conv_bias) atIndex:6];
        [block_encoder setBuffer:buffer_storage(residual) offset:buffer_offset(residual) atIndex:7];
        [block_encoder setBuffer:buffer_storage(output) offset:buffer_offset(output) atIndex:8];
        [block_encoder setBytes:&params length:sizeof(params) atIndex:9];
        const NSUInteger count = params.batch * params.channels * params.height * params.width;
        const NSUInteger threads = std::min<NSUInteger>(pipelines.block.maxTotalThreadsPerThreadgroup, 256);
        [block_encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [block_encoder endEncoding];
      }
    });
  }
  return output;
}

void register_resblock_op(pybind11::module_& module) {
  module.def(
      "fused_resblock_forward", &fused_resblock_forward,
      "Fused experimental ResBlock second half",
      pybind11::arg("input"), pybind11::arg("embedding"),
      pybind11::arg("norm_weight"), pybind11::arg("norm_bias"),
      pybind11::arg("conv_weight"), pybind11::arg("conv_bias"),
      pybind11::arg("residual"), pybind11::arg("groups"), pybind11::arg("epsilon"));
}