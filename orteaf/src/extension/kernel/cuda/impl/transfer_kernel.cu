#include <cuda_runtime.h>

#include <cstdint>

namespace {

inline constexpr std::uint8_t kTransferShapeInlineCapacity = 8;

struct TransferLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kTransferShapeInlineCapacity]{};
  std::int32_t src_strides[kTransferShapeInlineCapacity]{};
  std::int32_t dst_strides[kTransferShapeInlineCapacity]{};
};

__device__ inline std::int64_t physicalIndexForLinear(
    std::uint32_t linear, std::uint32_t rank, const std::uint32_t *shape,
    const std::int32_t *strides, std::uint32_t offset) {
  std::uint32_t remaining = linear;
  std::int64_t index = static_cast<std::int64_t>(offset);
  for (std::uint32_t d = rank; d-- > 0;) {
    const auto dim = shape[d];
    const auto coord = remaining % dim;
    remaining /= dim;
    index += static_cast<std::int64_t>(coord) *
             static_cast<std::int64_t>(strides[d]);
  }
  return index;
}

} // namespace

extern "C" __global__ void orteaf_copy_contiguous_to_strided_u8(
    const std::uint8_t *input, std::uint8_t *output, std::uint32_t input_offset,
    std::uint32_t output_offset, std::uint32_t numel, std::uint32_t elem_size,
    TransferLayoutParams layout) {
  const auto idx =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= numel) {
    return;
  }

  const auto src_elem_index = static_cast<std::int64_t>(input_offset + idx);
  const auto dst_elem_index = physicalIndexForLinear(
      idx, layout.rank, layout.shape, layout.dst_strides, output_offset);

  const auto src_byte = src_elem_index * static_cast<std::int64_t>(elem_size);
  const auto dst_byte = dst_elem_index * static_cast<std::int64_t>(elem_size);
  for (std::uint32_t i = 0; i < elem_size; ++i) {
    output[dst_byte + static_cast<std::int64_t>(i)] =
        input[src_byte + static_cast<std::int64_t>(i)];
  }
}

extern "C" __global__ void orteaf_copy_strided_to_contiguous_u8(
    const std::uint8_t *input, std::uint8_t *output, std::uint32_t input_offset,
    std::uint32_t output_offset, std::uint32_t numel, std::uint32_t elem_size,
    TransferLayoutParams layout) {
  const auto idx =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= numel) {
    return;
  }

  const auto src_elem_index = physicalIndexForLinear(
      idx, layout.rank, layout.shape, layout.src_strides, input_offset);
  const auto dst_elem_index = static_cast<std::int64_t>(output_offset + idx);

  const auto src_byte = src_elem_index * static_cast<std::int64_t>(elem_size);
  const auto dst_byte = dst_elem_index * static_cast<std::int64_t>(elem_size);
  for (std::uint32_t i = 0; i < elem_size; ++i) {
    output[dst_byte + static_cast<std::int64_t>(i)] =
        input[src_byte + static_cast<std::int64_t>(i)];
  }
}
