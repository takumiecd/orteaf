#pragma once

#include <functional>

#include <orteaf/internal/kernel/storage/operand_id.h>
#include <orteaf/internal/kernel/storage/role.h>

namespace orteaf::internal::kernel {

/**
 * @brief Composite key for identifying a bound operand.
 *
 * Combines OperandId (kernel-side semantic) with Role (tensor-internal
 * role). This avoids OperandId explosion when a tensor has multiple storages.
 */
struct OperandKey {
  OperandId id{static_cast<OperandId>(0)};
  Role role{Role::Data};

  constexpr OperandKey() = default;
  constexpr OperandKey(OperandId id_in, Role role_in) noexcept
      : id(id_in), role(role_in) {}

  friend constexpr bool operator==(const OperandKey &lhs,
                                   const OperandKey &rhs) noexcept {
    return lhs.id == rhs.id && lhs.role == rhs.role;
  }

  friend constexpr bool operator!=(const OperandKey &lhs,
                                   const OperandKey &rhs) noexcept {
    return !(lhs == rhs);
  }
};

/// @brief Convenience helper for default role (Data).
constexpr OperandKey makeOperandKey(OperandId id, Role role = Role::Data) {
  return OperandKey{id, role};
}

} // namespace orteaf::internal::kernel

namespace std {
template <> struct hash<::orteaf::internal::kernel::OperandKey> {
  std::size_t operator()(const ::orteaf::internal::kernel::OperandKey &operand) const noexcept {
    std::size_t h1 = std::hash<::orteaf::internal::kernel::OperandId>{}(operand.id);
    std::size_t h2 = std::hash<::orteaf::internal::kernel::Role>{}(operand.role);
    constexpr std::size_t kHashMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std
