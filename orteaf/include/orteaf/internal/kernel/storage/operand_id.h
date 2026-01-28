#pragma once

#include <cstdint>
#include <functional>

namespace orteaf::internal::kernel {

/**
 * @brief Operand ID enum for kernel storage binding identification.
 *
 * A strongly-typed enum class based on std::uint64_t for kernel storage
 * binding identification. Provides type safety for use as operand identifiers.
 * Values are defined in the auto-generated operand_id.def file.
 *
 * Each OperandId has associated metadata including default access pattern
 * and description, available through the generated operand_id_tables.h.
 */
enum class OperandId : std::uint64_t {
#define OPERAND_ID(name, value) name = value,
#include <orteaf/kernel/operand_id.def>
#undef OPERAND_ID
};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {
template <> struct hash<::orteaf::internal::kernel::OperandId> {
  std::size_t operator()(
      const ::orteaf::internal::kernel::OperandId &operand_id) const noexcept {
    return std::hash<std::uint64_t>{}(static_cast<std::uint64_t>(operand_id));
  }
};
} // namespace std
