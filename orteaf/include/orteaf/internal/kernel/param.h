#pragma once

#include <functional>
#include <orteaf/internal/kernel/array_view.h>
#include <orteaf/internal/kernel/param_id.h>
#include <orteaf/kernel/param_id_tables.h>
#include <variant>

namespace orteaf::internal::kernel {

/**
 * @brief Type-erased parameter combining ParamId with its typed value.
 *
 * Param associates a ParamId with a value from the auto-generated ParamValue
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
   * @brief Construct a parameter with a float value.
   */
  Param(ParamId id, float value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Construct a parameter with a double value.
   */
  Param(ParamId id, double value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Construct a parameter with an int value.
   */
  Param(ParamId id, int value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Construct a parameter with a size_t value.
   */
  Param(ParamId id, std::size_t value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Construct a parameter with a void* value.
   */
  Param(ParamId id, void *value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Construct a parameter with an ArrayView value.
   */
  template <typename T>
  Param(ParamId id, ArrayView<T> value) noexcept : id_(id), value_(value) {}

  /**
   * @brief Get the parameter ID.
   */
  constexpr ParamId id() const noexcept { return id_; }

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
    return lhs.id_ == rhs.id_ && lhs.value_ == rhs.value_;
  }

  /**
   * @brief Inequality comparison.
   */
  friend bool operator!=(const Param &lhs, const Param &rhs) noexcept {
    return !(lhs == rhs);
  }

private:
  ParamId id_;
  Value value_;
};

} // namespace orteaf::internal::kernel

// Hash support for std::unordered_map and std::unordered_set
namespace std {

template <> struct hash<::orteaf::internal::kernel::Param> {
  std::size_t
  operator()(const ::orteaf::internal::kernel::Param &param) const noexcept {
    // Combine hash of id and value
    std::size_t h1 =
        std::hash<::orteaf::internal::kernel::ParamId>{}(param.id());
    std::size_t h2 = std::visit(
        [](const auto &v) { return std::hash<std::decay_t<decltype(v)>>{}(v); },
        param.value());
    // Simple hash combination
    return h1 ^ (h2 << 1);
  }
};

} // namespace std
