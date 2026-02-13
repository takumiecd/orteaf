#pragma once

#include <cstdint>

namespace orteaf::extension::kernel::common::layout {

inline constexpr std::uint8_t kShapeInlineCapacity = 8;

struct FillLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kShapeInlineCapacity]{};
  std::int32_t strides[kShapeInlineCapacity]{};
};

struct TransferLayoutParams {
  std::uint32_t rank{};
  std::uint32_t shape[kShapeInlineCapacity]{};
  std::int32_t src_strides[kShapeInlineCapacity]{};
  std::int32_t dst_strides[kShapeInlineCapacity]{};
};

static_assert(sizeof(FillLayoutParams) == 68);
static_assert(sizeof(TransferLayoutParams) == 100);
static_assert(alignof(FillLayoutParams) == 4);
static_assert(alignof(TransferLayoutParams) == 4);

} // namespace orteaf::extension::kernel::common::layout
