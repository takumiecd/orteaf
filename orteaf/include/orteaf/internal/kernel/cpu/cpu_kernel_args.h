#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/param.h>

#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/kernel/cpu/cpu_storage_binding.h>
#include <orteaf/internal/kernel/kernel_key.h>
#include <orteaf/internal/kernel/storage_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU kernel arguments container.
 *
 * Manages storage bindings and parameters for CPU kernel execution.
 * Uses structured storage bindings with StorageId for type-safe management.
 */
class CpuKernelArgs {
public:
  using StorageLease = ::orteaf::internal::storage::CpuStorageLease;

  static constexpr std::size_t kMaxBindings = 16;
  static constexpr std::size_t kMaxParams = 16;

  /**
   * @brief Add a storage binding with the specified ID.
   *
   * @param id Storage identifier (contains access pattern metadata)
   * @param lease Storage lease to bind
   */
  void addStorage(StorageId id, StorageLease lease) {
    if (storages_.size() >= kMaxBindings) {
      return;
    }
    storages_.pushBack(CpuStorageBinding{id, std::move(lease)});
  }

  /**
   * @brief Find a storage binding by ID.
   *
   * @param id Storage identifier to search for
   * @return Pointer to CpuStorageBinding if found, nullptr otherwise
   */
  const CpuStorageBinding *findStorage(StorageId id) const {
    for (const auto &binding : storages_) {
      if (binding.id == id) {
        return &binding;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a storage binding by ID (mutable version).
   */
  CpuStorageBinding *findStorage(StorageId id) {
    for (auto &binding : storages_) {
      if (binding.id == id) {
        return &binding;
      }
    }
    return nullptr;
  }

  /**
   * @brief Get the list of all storage bindings.
   */
  const auto &storageList() const { return storages_; }

  /**
   * @brief Get the number of storage bindings.
   */
  std::size_t storageCount() const { return storages_.size(); }

  /**
   * @brief Get the maximum number of storage bindings.
   */
  std::size_t storageCapacity() const { return kMaxBindings; }

  /**
   * @brief Clear all storage bindings.
   */
  void clearStorages() { storages_.clear(); }

  /**
   * @brief Add a parameter.
   */
  void addParam(Param param) { params_.pushBack(std::move(param)); }

  /**
   * @brief Find a parameter by ID.
   */
  const Param *findParam(ParamId id) const {
    for (const auto &p : params_) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Find a parameter by ID (mutable version).
   */
  Param *findParam(ParamId id) {
    for (auto &p : params_) {
      if (p.id() == id) {
        return &p;
      }
    }
    return nullptr;
  }

  /**
   * @brief Get the list of all parameters.
   */
  const auto &paramList() const { return params_; }

  /**
   * @brief Clear all parameters.
   */
  void clearParams() { params_.clear(); }

private:
  ::orteaf::internal::base::SmallVector<CpuStorageBinding, kMaxBindings>
      storages_{};
  ::orteaf::internal::base::SmallVector<Param, kMaxParams> params_{};
};

} // namespace orteaf::internal::kernel::cpu
