#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Role of a storage within a tensor or kernel binding.
 *
 * Roles distinguish multiple storages within the same logical tensor
 * (e.g., data vs. indices). The default role is Data.
 */
enum class StorageRole : std::uint8_t {
  Data = 0,
  Index = 1,
  Indptr = 2,
  Meta = 3,
};

} // namespace orteaf::internal::kernel

namespace std {
template <> struct hash<::orteaf::internal::kernel::StorageRole> {
  std::size_t operator()(const ::orteaf::internal::kernel::StorageRole &role) const noexcept {
    return std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(role));
  }
};
} // namespace std
