#pragma once

#include <optional>

#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_key.h>

namespace orteaf::internal::kernel {

/**
 * @brief Composite key for identifying a parameter.
 *
 * Params may be global (no operand scope) or scoped to a specific operand key
 * (e.g., shape/strides for Input0/Data).
 */
struct ParamKey {
  ParamId id{static_cast<ParamId>(0)};
  std::optional<OperandKey> operand_key{};

  constexpr ParamKey() = default;
  constexpr ParamKey(ParamId id_in,
                     std::optional<OperandKey> operand_key_in = std::nullopt) noexcept
      : id(id_in), operand_key(operand_key_in) {}

  static constexpr ParamKey global(ParamId id_in) noexcept {
    return ParamKey{id_in, std::nullopt};
  }

  static constexpr ParamKey scoped(ParamId id_in, OperandKey operand_key) noexcept {
    return ParamKey{id_in, operand_key};
  }

  friend constexpr bool operator==(const ParamKey &lhs,
                                   const ParamKey &rhs) noexcept {
    return lhs.id == rhs.id && lhs.operand_key == rhs.operand_key;
  }

  friend constexpr bool operator!=(const ParamKey &lhs,
                                   const ParamKey &rhs) noexcept {
    return !(lhs == rhs);
  }
};

} // namespace orteaf::internal::kernel

namespace std {
template <> struct hash<::orteaf::internal::kernel::ParamKey> {
  std::size_t operator()(const ::orteaf::internal::kernel::ParamKey &key) const noexcept {
    std::size_t h1 = std::hash<::orteaf::internal::kernel::ParamId>{}(key.id);
    std::size_t h2 = 0;
    if (key.operand_key.has_value()) {
      h2 = std::hash<::orteaf::internal::kernel::OperandKey>{}(*key.operand_key);
    }
    constexpr std::size_t kHashMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std
