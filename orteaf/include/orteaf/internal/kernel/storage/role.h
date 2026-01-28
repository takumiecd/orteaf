#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Role of a storage within an operand or kernel binding.
 *
 * Roles distinguish multiple storages within the same logical tensor
 * (e.g., data vs. indices). The default role is Data.
 */
enum class Role : std::uint8_t {
#define ROLE(name, value) name = value,
#include <orteaf/kernel/role.def>
#undef ROLE
};

} // namespace orteaf::internal::kernel

namespace std {
template <> struct hash<::orteaf::internal::kernel::Role> {
  std::size_t operator()(const ::orteaf::internal::kernel::Role &role) const noexcept {
    return std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(role));
  }
};
} // namespace std
