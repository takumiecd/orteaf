#pragma once

#include <functional>
#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/array_view.h>
#include <orteaf/internal/kernel/param_id.h>
#include <orteaf/kernel/param_id_tables.h>
#include <variant>

namespace orteaf::internal::kernel {

// Default inline capacity for parameter lists
inline constexpr std::size_t kDefaultParamListCapacity = 16;

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

/**
 * @brief Parameter list container.
 *
 * Manages a collection of parameters with efficient inline storage.
 * Provides convenient methods for adding and finding parameters by ID.
 * Used across all kernel argument implementations.
 */
class ParamList {
public:
  using Storage =
      ::orteaf::internal::base::SmallVector<Param, kDefaultParamListCapacity>;
  using iterator = typename Storage::iterator;
  using const_iterator = typename Storage::const_iterator;

  /**
   * @brief Default constructor.
   */
  ParamList() = default;

  /**
   * @brief Add a parameter to the list.
   */
  void add(Param param) { storage_.pushBack(std::move(param)); }

  /**
   * @brief Add a parameter to the list (alias for compatibility).
   */
  void pushBack(Param param) { storage_.pushBack(std::move(param)); }

  /**
   * @brief Find a parameter by ID.
   *
   * @param id Parameter identifier to search for
   * @return Pointer to Param if found, nullptr otherwise
   */
  const Param *find(ParamId id) const {
    for (const auto &p : storage_) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a parameter by ID (mutable version).
   */
  Param *find(ParamId id) {
    for (auto &p : storage_) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Get the number of parameters.
   */
  std::size_t size() const { return storage_.size(); }

  /**
   * @brief Check if the list is empty.
   */
  bool empty() const { return storage_.empty(); }

  /**
   * @brief Get the current capacity.
   */
  std::size_t capacity() const { return storage_.capacity(); }

  /**
   * @brief Clear all parameters.
   */
  void clear() { storage_.clear(); }

  /**
   * @brief Get iterator to the beginning.
   */
  iterator begin() { return storage_.begin(); }
  const_iterator begin() const { return storage_.begin(); }

  /**
   * @brief Get iterator to the end.
   */
  iterator end() { return storage_.end(); }
  const_iterator end() const { return storage_.end(); }

  /**
   * @brief Access underlying storage.
   */
  const Storage &storage() const { return storage_; }
  Storage &storage() { return storage_; }

private:
  Storage storage_{};
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
    // Robust hash combination (boost::hash_combine style).
    constexpr std::size_t kHashMix =
        static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    std::size_t seed = h1;
    seed ^= h2 + kHashMix + (seed << 6) + (seed >> 2);
    return seed;
  }
};

} // namespace std
