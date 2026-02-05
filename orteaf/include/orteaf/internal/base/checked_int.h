#pragma once

#include <cstdint>
#include <limits>

namespace orteaf::internal::base {

inline bool addOverflowI64(std::int64_t lhs, std::int64_t rhs,
                           std::int64_t &out) {
  if (rhs > 0 &&
      lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    return true;
  }
  if (rhs < 0 &&
      lhs < std::numeric_limits<std::int64_t>::min() - rhs) {
    return true;
  }
  out = lhs + rhs;
  return false;
}

inline bool absToU64(std::int64_t value, std::uint64_t &out) {
  if (value == std::numeric_limits<std::int64_t>::min()) {
    out = static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max()) +
          1u;
    return true;
  }
  out = static_cast<std::uint64_t>(value < 0 ? -value : value);
  return false;
}

inline bool mulOverflowI64(std::int64_t lhs, std::int64_t rhs,
                           std::int64_t &out) {
  std::uint64_t abs_lhs = 0;
  std::uint64_t abs_rhs = 0;
  absToU64(lhs, abs_lhs);
  absToU64(rhs, abs_rhs);

  if (abs_lhs == 0 || abs_rhs == 0) {
    out = 0;
    return false;
  }

  if (abs_lhs >
      std::numeric_limits<std::uint64_t>::max() / abs_rhs) {
    return true;
  }

  const std::uint64_t prod = abs_lhs * abs_rhs;
  const bool negative = (lhs < 0) ^ (rhs < 0);
  if (negative) {
    const std::uint64_t limit =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()) +
        1u;
    if (prod > limit) {
      return true;
    }
    if (prod == limit) {
      out = std::numeric_limits<std::int64_t>::min();
      return false;
    }
    out = -static_cast<std::int64_t>(prod);
    return false;
  }

  if (prod >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max())) {
    return true;
  }
  out = static_cast<std::int64_t>(prod);
  return false;
}

}  // namespace orteaf::internal::base
