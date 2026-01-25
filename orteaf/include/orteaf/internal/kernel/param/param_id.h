#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Parameter ID enum for kernel parameter identification.
 *
 * A strongly-typed enum class based on std::uint64_t for kernel parameter
 * identification. Provides type safety for use as parameter identifiers.
 * Values are defined in the auto-generated param_id.def file.
 */
enum class ParamId : std::uint64_t {
#define PARAM_ID(name, value) name = value,
#include <orteaf/kernel/param_id.def>
#undef PARAM_ID
};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::ParamId> {
  std::size_t operator()(
      const ::orteaf::internal::kernel::ParamId &param_id) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(param_id));
  }
};
} // namespace std
