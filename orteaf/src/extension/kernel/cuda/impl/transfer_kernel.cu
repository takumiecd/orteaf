#include <cuda_runtime.h>

#include <cstddef>
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
    if (dim == 0) {
      return index;
    }
    const auto coord = remaining % dim;
    remaining /= dim;
    index += static_cast<std::int64_t>(coord) *
             static_cast<std::int64_t>(strides[d]);
  }
  return index;
}

__device__ inline void copyElementBytes(const std::uint8_t *src,
                                        std::uint8_t *dst,
                                        std::uint32_t elem_size) {
  std::uint32_t copied = 0;
  const auto src_addr = reinterpret_cast<std::uintptr_t>(src);
  const auto dst_addr = reinterpret_cast<std::uintptr_t>(dst);
  if (((src_addr | dst_addr) & 0x7U) == 0U) {
    const auto chunks8 = elem_size / 8U;
    const auto *src_u64 = reinterpret_cast<const std::uint64_t *>(src);
    auto *dst_u64 = reinterpret_cast<std::uint64_t *>(dst);
    for (std::uint32_t i = 0; i < chunks8; ++i) {
      dst_u64[i] = src_u64[i];
    }
    copied = chunks8 * 8U;
  } else if (((src_addr | dst_addr) & 0x3U) == 0U) {
    const auto chunks4 = elem_size / 4U;
    const auto *src_u32 = reinterpret_cast<const std::uint32_t *>(src);
    auto *dst_u32 = reinterpret_cast<std::uint32_t *>(dst);
    for (std::uint32_t i = 0; i < chunks4; ++i) {
      dst_u32[i] = src_u32[i];
    }
    copied = chunks4 * 4U;
  }
  for (std::uint32_t i = copied; i < elem_size; ++i) {
    dst[i] = src[i];
  }
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

  const auto src =
      input + src_elem_index * static_cast<std::int64_t>(elem_size);
  auto *dst = output + dst_elem_index * static_cast<std::int64_t>(elem_size);
  copyElementBytes(src, dst, elem_size);
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

  const auto *src =
      input + src_elem_index * static_cast<std::int64_t>(elem_size);
  auto *dst = output + dst_elem_index * static_cast<std::int64_t>(elem_size);
  copyElementBytes(src, dst, elem_size);
}
