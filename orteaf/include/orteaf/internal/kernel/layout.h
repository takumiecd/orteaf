#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Layout enum for memory layout pattern identification.
 *
 * A strongly-typed enum class based on std::uint64_t for memory layout
 * identification. Represents different memory layout patterns such as
 * row-major, column-major, etc.
 */
enum class Layout : std::uint64_t {};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::Layout> {
  std::size_t
  operator()(const ::orteaf::internal::kernel::Layout &layout) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(layout));
  }
};
} // namespace std
