#include <cuda_runtime.h>

#include <cstdint>

namespace {

inline constexpr std::uint32_t kFillShapeInlineCapacity = 8;

struct FillLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kFillShapeInlineCapacity]{};
  std::int32_t strides[kFillShapeInlineCapacity]{};
};
// ABI contract: must match common::layout::FillLayoutParams on host side.

} // namespace

extern "C" __global__ void
orteaf_fill_strided_f32(float *output, std::uint32_t offset,
                        std::uint32_t numel, float value,
                        FillLayoutParams layout) {
  const auto idx = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= numel) {
    return;
  }

  std::uint32_t linear = idx;
  std::int64_t flat_index = static_cast<std::int64_t>(offset);
  for (std::uint32_t d = layout.rank; d-- > 0;) {
    const auto dim = layout.shape[d];
    if (dim == 0) {
      return;
    }
    const auto coord = linear % dim;
    linear /= dim;
    flat_index += static_cast<std::int64_t>(coord) *
                  static_cast<std::int64_t>(layout.strides[d]);
  }

  output[flat_index] = value;
}
