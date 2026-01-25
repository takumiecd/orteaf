#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Storage binding ID enum for kernel storage identification.
 *
 * A strongly-typed enum class based on std::uint64_t for kernel storage
 * binding identification. Provides type safety for use as storage identifiers.
 * Values are defined in the auto-generated storage_id.def file.
 *
 * Each StorageId has associated metadata including default access pattern
 * and description, available through the generated storage_id_tables.h.
 */
enum class StorageId : std::uint64_t {
#define STORAGE_ID(name, value) name = value,
#include <orteaf/kernel/storage_id.def>
#undef STORAGE_ID
};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::StorageId> {
  std::size_t operator()(
      const ::orteaf::internal::kernel::StorageId &storage_id) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(storage_id));
  }
};
} // namespace std
