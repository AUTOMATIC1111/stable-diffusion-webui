#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

namespace {

struct QuantizedLinearParams {
  uint32_t rows;
  uint32_t input_features;
  uint32_t output_features;
};

static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static inline size_t getMTLBufferOffset(const at::Tensor& tensor) {
  return tensor.storage_offset() * tensor.element_size();
}

static id<MTLComputePipelineState> getQuantizedLinearPipeline() {
  static id<MTLComputePipelineState> pipeline = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSString* source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct QuantizedLinearParams {
  uint rows;
  uint input_features;
  uint output_features;
};

kernel void quantized_linear_int8_half(
    device const half* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device const half* scale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.output_features;
  if (index >= count) {
    return;
  }

  const uint row = index / params.output_features;
  const uint column = index - row * params.output_features;
  const uint input_base = row * params.input_features;
  const uint weight_base = column * params.input_features;
  float sum = 0.0f;
  for (uint feature = 0; feature < params.input_features; ++feature) {
    sum += float(input[input_base + feature]) * float(weight[weight_base + feature]);
  }

  output[index] = half(sum * float(scale[column]));
}
)METAL";

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(
        library != nil,
        "Failed to compile quantized linear Metal library: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
    id<MTLFunction> function = [library newFunctionWithName:@"quantized_linear_int8_half"];
    TORCH_CHECK(function != nil, "Quantized linear Metal function was not found");
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(
        pipeline != nil,
        "Failed to create quantized linear Metal pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
  });
  return pipeline;
}

static id<MTLComputePipelineState> getQuantizedLinearSIMDPipeline() {
  static id<MTLComputePipelineState> pipeline = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSString* source = @R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct QuantizedLinearParams {
  uint rows;
  uint input_features;
  uint output_features;
};

kernel void quantized_linear_int8_simd_half(
    device const half* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device const half* scale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantizedLinearParams& params [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
  threadgroup half input_tile[64];
  threadgroup half weight_tile[64];
  threadgroup float output_tile[64];

  const uint output_column = tile.x * 8;
  const uint output_row = tile.y * 8;
  simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint feature_base = 0; feature_base < params.input_features; feature_base += 8) {
    for (uint element = lane; element < 64; element += 32) {
      const uint local_row = element / 8;
      const uint local_feature = element - local_row * 8;
      input_tile[element] = input[
          (output_row + local_row) * params.input_features + feature_base + local_feature];

      const uint local_output = element % 8;
      const uint feature = element / 8;
      const uint weight_output = output_column + local_output;
      weight_tile[element] = half(weight[
          weight_output * params.input_features + feature_base + feature]) * scale[weight_output];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 input_matrix;
    simdgroup_half8x8 weight_matrix;
    simdgroup_load(input_matrix, input_tile, 8);
    simdgroup_load(weight_matrix, weight_tile, 8);
    simdgroup_multiply_accumulate(accumulator, input_matrix, weight_matrix, accumulator);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(accumulator, output_tile, 8);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint element = lane; element < 64; element += 32) {
    const uint local_row = element / 8;
    const uint local_output = element - local_row * 8;
    output[(output_row + local_row) * params.output_features + output_column + local_output] =
        half(output_tile[element]);
  }
}
)METAL";

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    TORCH_CHECK(
        library != nil,
        "Failed to compile SIMD quantized linear Metal library: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
    id<MTLFunction> function = [library newFunctionWithName:@"quantized_linear_int8_simd_half"];
    TORCH_CHECK(function != nil, "SIMD quantized linear Metal function was not found");
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(
        pipeline != nil,
        "Failed to create SIMD quantized linear Metal pipeline: ",
        error ? [[error localizedDescription] UTF8String] : "unknown error");
  });
  return pipeline;
}

} // namespace

torch::Tensor quantized_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale) {
  TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(weight.device().is_mps() && scale.device().is_mps(), "weights must be MPS tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be float16");
  TORCH_CHECK(input.dim() == 2, "input must have shape [rows, input_features]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [output_features, input_features]");
  TORCH_CHECK(scale.dim() == 1, "scale must have shape [output_features]");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.size(1) == weight.size(1), "input and weight feature dimensions must match");
  TORCH_CHECK(scale.size(0) == weight.size(0), "scale and weight output dimensions must match");

  auto output = torch::empty({input.size(0), weight.size(0)}, input.options());
  QuantizedLinearParams params = {
      static_cast<uint32_t>(input.size(0)),
      static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(weight.size(0)),
  };
  const uint32_t count = params.rows * params.output_features;
  auto pipeline = getQuantizedLinearPipeline();

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        at::mps::getCurrentMPSStream()->endKernelCoalescing();
        id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:getMTLBufferStorage(input) offset:getMTLBufferOffset(input) atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(weight) offset:getMTLBufferOffset(weight) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(scale) offset:getMTLBufferOffset(scale) atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output) offset:getMTLBufferOffset(output) atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        const NSUInteger threads = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
      }
    });
  }
  return output;
}

torch::Tensor quantized_linear_simd_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale) {
  TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(weight.device().is_mps() && scale.device().is_mps(), "weights must be MPS tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be float16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && scale.dim() == 1, "invalid tensor dimensions");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.size(1) == weight.size(1), "input and weight feature dimensions must match");
  TORCH_CHECK(scale.size(0) == weight.size(0), "scale and weight output dimensions must match");
  TORCH_CHECK(input.size(0) % 8 == 0, "input rows must be divisible by 8");
  TORCH_CHECK(input.size(1) % 8 == 0, "input features must be divisible by 8");
  TORCH_CHECK(weight.size(0) % 8 == 0, "output features must be divisible by 8");

  auto output = torch::empty({input.size(0), weight.size(0)}, input.options());
  QuantizedLinearParams params = {
      static_cast<uint32_t>(input.size(0)),
      static_cast<uint32_t>(input.size(1)),
      static_cast<uint32_t>(weight.size(0)),
  };
  auto pipeline = getQuantizedLinearSIMDPipeline();

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        at::mps::getCurrentMPSStream()->endKernelCoalescing();
        id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:getMTLBufferStorage(input) offset:getMTLBufferOffset(input) atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(weight) offset:getMTLBufferOffset(weight) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(scale) offset:getMTLBufferOffset(scale) atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output) offset:getMTLBufferOffset(output) atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(params.output_features / 8, params.rows / 8, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
      }
    });
  }
  return output;
}

void register_quantized_linear_op(pybind11::module_& module) {
  module.def(
      "quantized_linear_forward",
      &quantized_linear_forward,
      "Direct Metal int8-weight linear forward pass",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("scale"));
  module.def(
      "quantized_linear_simd_forward",
      &quantized_linear_simd_forward,
      "Tiled SIMD-group Metal int8-weight linear forward pass",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("scale"));
}
