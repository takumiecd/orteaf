#pragma once

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/storage/operand_key.h>

#include <cstddef>

namespace orteaf::internal::kernel {

// Default inline capacity for storage lists
inline constexpr std::size_t kDefaultStorageListCapacity = 16;

/**
 * @brief Generic storage list container.
 *
 * Manages a collection of storage bindings with efficient inline storage.
 * Provides convenient methods for adding and finding storages by operand key.
 * Template parameter allows use with different storage binding types
 * (e.g., StorageBinding).
 *
 * @tparam StorageBinding The storage binding type (must have a 'key' field)
 * @tparam InlineCapacity Number of inline storage slots (default: 16)
 *
 * Example:
 * @code
 * using AnyBinding = StorageBinding;
 * StorageList<AnyBinding> storages;
 * storages.add(AnyBinding{makeOperandKey(OperandId::Input0), lease});
 * auto* binding = storages.find(makeOperandKey(OperandId::Input0));
 * @endcode
 */
template <typename StorageBinding,
          std::size_t InlineCapacity = kDefaultStorageListCapacity>
class StorageList {
public:
  using Storage = ::orteaf::internal::base::SmallVector<StorageBinding, InlineCapacity>;
  using iterator = typename Storage::iterator;
  using const_iterator = typename Storage::const_iterator;

  /**
   * @brief Default constructor.
   */
  StorageList() = default;

  /**
   * @brief Add a storage binding to the list.
   */
  void add(StorageBinding binding) { storage_.pushBack(std::move(binding)); }

  /**
   * @brief Add a storage binding to the list (alias for compatibility).
   */
  void pushBack(StorageBinding binding) { storage_.pushBack(std::move(binding)); }

  /**
   * @brief Find a storage binding by operand key.
   *
   * @param key Operand key to search for
   * @return Pointer to StorageBinding if found, nullptr otherwise
   */
  const StorageBinding *find(OperandKey key) const {
    for (const auto &binding : storage_) {
      if (binding.key == key) {
        return &binding;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a storage binding by operand key (mutable version).
   */
  StorageBinding *find(OperandKey key) {
    for (auto &binding : storage_) {
      if (binding.key == key) {
        return &binding;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a storage binding by ID (defaults to Data role).
   */
  const StorageBinding *find(OperandId id) const {
    return find(makeOperandKey(id));
  }

  /**
   * @brief Find a storage binding by ID (mutable, defaults to Data role).
   */
  StorageBinding *find(OperandId id) { return find(makeOperandKey(id)); }

  /**
   * @brief Get the number of storage bindings.
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
   * @brief Clear all storage bindings.
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
