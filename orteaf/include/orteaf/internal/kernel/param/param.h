#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <variant>

#include <orteaf/internal/base/array_view.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/param_key.h>
#include <orteaf/kernel/param_id_tables.h>

namespace orteaf::internal::kernel {

/**
 * @brief Type-erased parameter combining ParamKey with its typed value.
 *
 * Param associates a ParamKey with a value from the auto-generated ParamValue
 * variant. This allows heterogeneous collections of parameters (e.g.,
 * std::vector<Param>) while maintaining type safety through std::variant.
 *
 * All value types are POD or trivially copyable, making Param suitable for
 * passing to CUDA/Metal kernels.
 */
class Param {
public:
  using Value = ::orteaf::generated::param_id_tables::ParamValue;

  /**
   * @brief Construct a parameter with a float value (global).
   */
  Param(ParamId id, float value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with a double value (global).
   */
  Param(ParamId id, double value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with an int value (global).
   */
  Param(ParamId id, int value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with an int64_t value (global).
   */
  Param(ParamId id, std::int64_t value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with a size_t value (global).
   */
  Param(ParamId id, std::size_t value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with a uint32_t value (global).
   */
  Param(ParamId id, std::uint32_t value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with a void* value (global).
   */
  Param(ParamId id, void *value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with an ArrayView value (global).
   */
  template <typename T>
  Param(ParamId id, internal::base::ArrayView<T> value) noexcept
      : key_(ParamKey::global(id)), value_(value) {}

  /**
   * @brief Construct a parameter with a float value (scoped).
   */
  Param(ParamKey key, float value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with a double value (scoped).
   */
  Param(ParamKey key, double value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with an int value (scoped).
   */
  Param(ParamKey key, int value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with an int64_t value (scoped).
   */
  Param(ParamKey key, std::int64_t value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with a size_t value (scoped).
   */
  Param(ParamKey key, std::size_t value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with a uint32_t value (scoped).
   */
  Param(ParamKey key, std::uint32_t value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with a void* value (scoped).
   */
  Param(ParamKey key, void *value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Construct a parameter with an ArrayView value (scoped).
   */
  template <typename T>
  Param(ParamKey key, internal::base::ArrayView<T> value) noexcept : key_(key), value_(value) {}

  /**
   * @brief Get the parameter ID.
   */
  constexpr ParamId id() const noexcept { return key_.id; }

  /**
   * @brief Get the parameter key.
   */
  constexpr const ParamKey &key() const noexcept { return key_; }

  /**
   * @brief Get the parameter value (const reference).
   */
  constexpr const Value &value() const noexcept { return value_; }

  /**
   * @brief Get the parameter value (mutable reference).
   */
  constexpr Value &value() noexcept { return value_; }

  /**
   * @brief Try to get the value as a specific type.
   *
   * @tparam T The type to retrieve
   * @return Pointer to the value if types match, nullptr otherwise
   */
  template <typename T> const T *tryGet() const noexcept {
    return std::get_if<T>(&value_);
  }

  /**
   * @brief Try to get the value as a specific type (mutable).
   *
   * @tparam T The type to retrieve
   * @return Pointer to the value if types match, nullptr otherwise
   */
  template <typename T> T *tryGet() noexcept { return std::get_if<T>(&value_); }

  /**
   * @brief Visit the value with a visitor callable.
   *
   * @tparam Visitor Callable accepting any variant type
   * @return Result of the visitor
   */
  template <typename Visitor> auto visit(Visitor &&visitor) const {
    return std::visit(std::forward<Visitor>(visitor), value_);
  }

  /**
   * @brief Visit the value with a visitor callable (mutable).
   *
   * @tparam Visitor Callable accepting any variant type
   * @return Result of the visitor
   */
  template <typename Visitor> auto visit(Visitor &&visitor) {
    return std::visit(std::forward<Visitor>(visitor), value_);
  }

  /**
   * @brief Equality comparison.
   */
  friend bool operator==(const Param &lhs, const Param &rhs) noexcept {
    return lhs.key_ == rhs.key_ && lhs.value_ == rhs.value_;
  }

  /**
   * @brief Inequality comparison.
   */
  friend bool operator!=(const Param &lhs, const Param &rhs) noexcept {
    return !(lhs == rhs);
  }

private:
  ParamKey key_;
  Value value_;
};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {

template <> struct hash<::orteaf::internal::kernel::Param> {
  std::size_t
  operator()(const ::orteaf::internal::kernel::Param &param) const noexcept {
    // Combine hash of key and value
    std::size_t h1 =
        std::hash<::orteaf::internal::kernel::ParamKey>{}(param.key());
    std::size_t h2 = std::visit(
        [](const auto &v) { return std::hash<std::decay_t<decltype(v)>>{}(v); },
        param.value());
    // Robust hash combination (boost::hash_combine style).
    constexpr std::size_t kHashMix =
        static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};

} // namespace std
