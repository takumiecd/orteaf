#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Variant enum for kernel optimization levels.
 *
 * A strongly-typed enum class based on std::uint64_t for kernel variant
 * identification. Represents different optimization levels or implementations
 * of the same operation.
 */
enum class Variant : std::uint64_t {};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::Variant> {
  std::size_t operator()(
      const ::orteaf::internal::kernel::Variant &variant) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(variant));
  }
};
} // namespace std
