#pragma once

#include <cstddef>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/param/param.h>

namespace orteaf::internal::kernel {

// Default inline capacity for parameter lists
inline constexpr std::size_t kDefaultParamListCapacity = 16;

/**
 * @brief Parameter list container.
 *
 * Manages a collection of parameters with efficient inline storage.
 * Provides convenient methods for adding and finding parameters by key/ID.
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
   * @brief Find a parameter by key.
   *
   * @param key Parameter key to search for
   * @return Pointer to Param if found, nullptr otherwise
   */
  const Param *find(ParamKey key) const {
    for (const auto &p : storage_) {
      if (p.key() == key) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a parameter by key (mutable version).
   */
  Param *find(ParamKey key) {
    for (auto &p : storage_) {
      if (p.key() == key) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a parameter by ID (global scope only).
   *
   * @param id Parameter identifier to search for
   * @return Pointer to Param if found, nullptr otherwise
   */
  const Param *find(ParamId id) const {
    return find(ParamKey::global(id));
  }

  /**
   * @brief Find a parameter by ID (mutable, global scope only).
   */
  Param *find(ParamId id) {
    return find(ParamKey::global(id));
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
