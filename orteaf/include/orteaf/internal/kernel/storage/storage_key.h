#pragma once

#include <functional>

#include <orteaf/internal/kernel/storage/storage_id.h>
#include <orteaf/internal/kernel/storage/storage_role.h>

namespace orteaf::internal::kernel {

/**
 * @brief Composite key for identifying a bound storage.
 *
 * Combines StorageId (kernel-side semantic) with StorageRole (tensor-internal
 * role). This avoids StorageId explosion when a tensor has multiple storages.
 */
struct StorageKey {
  StorageId id{static_cast<StorageId>(0)};
  StorageRole role{StorageRole::Data};

  constexpr StorageKey() = default;
  constexpr StorageKey(StorageId id_in, StorageRole role_in) noexcept
      : id(id_in), role(role_in) {}

  friend constexpr bool operator==(const StorageKey &lhs,
                                   const StorageKey &rhs) noexcept {
    return lhs.id == rhs.id && lhs.role == rhs.role;
  }

  friend constexpr bool operator!=(const StorageKey &lhs,
                                   const StorageKey &rhs) noexcept {
    return !(lhs == rhs);
  }
};

/// @brief Convenience helper for default role (Data).
constexpr StorageKey makeStorageKey(StorageId id,
                                    StorageRole role = StorageRole::Data) {
  return StorageKey{id, role};
}

} // namespace orteaf::internal::kernel

namespace std {
template <> struct hash<::orteaf::internal::kernel::StorageKey> {
  std::size_t operator()(const ::orteaf::internal::kernel::StorageKey &key) const noexcept {
    std::size_t h1 = std::hash<::orteaf::internal::kernel::StorageId>{}(key.id);
    std::size_t h2 = std::hash<::orteaf::internal::kernel::StorageRole>{}(key.role);
    constexpr std::size_t kHashMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std
