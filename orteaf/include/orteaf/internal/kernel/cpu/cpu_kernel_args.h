#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

#include <orteaf/internal/kernel/param/param_list.h>
#include <orteaf/internal/kernel/storage/storage_list.h>

#include <orteaf/internal/execution_context/cpu/context.h>
#include <orteaf/internal/kernel/core/access.h>
#include <orteaf/internal/kernel/cpu/cpu_storage_binding.h>
#include <orteaf/internal/kernel/core/kernel_key.h>
#include <orteaf/internal/kernel/storage/storage_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU kernel arguments container.
 *
 * Manages storage bindings and parameters for CPU kernel execution.
 * Uses structured storage bindings with StorageId for type-safe management.
 * Holds execution context for device and resource management.
 */
class CpuKernelArgs {
public:
  using Context = ::orteaf::internal::execution_context::cpu::Context;
  using StorageLease = ::orteaf::internal::storage::CpuStorageLease;
  using StorageListType = StorageList<CpuStorageBinding>;

  // Inline capacities; SmallVector can grow beyond these values.
  static constexpr std::size_t kMaxBindings = 16;
  static constexpr std::size_t kMaxParams = 16;

  /**
   * @brief Create kernel args from current execution context.
   */
  static CpuKernelArgs fromCurrentContext();

  /**
   * @brief Default constructor (uses current context).
   */
  CpuKernelArgs();

  /**
   * @brief Construct with explicit context.
   */
  explicit CpuKernelArgs(Context context);

  /**
   * @brief Tag type for no-init construction.
   *
   * Use this tag to construct CpuKernelArgs without initializing the context.
   * Primarily intended for testing purposes where full runtime setup is not
   * needed.
   */
  struct NoInit {};

  /**
   * @brief Construct without initializing context (for testing).
   *
   * Creates a CpuKernelArgs with an empty context. Useful for unit testing
   * kernel schemas without requiring full CPU runtime configuration.
   */
  explicit CpuKernelArgs(NoInit) noexcept {}

  /**
   * @brief Get the execution context.
   */
  const Context &context() const { return context_; }

  /**
   * @brief Get the execution context (mutable).
   */
  Context &context() { return context_; }

  /**
   * @brief Add a storage binding with the specified ID.
   *
   * @param id Storage identifier (contains access pattern metadata)
   * @param lease Storage lease to bind
   */
  void addStorage(StorageId id, StorageLease lease) {
    storages_.add(CpuStorageBinding{id, std::move(lease)});
  }

  /**
   * @brief Find a storage binding by ID.
   *
   * @param id Storage identifier to search for
   * @return Pointer to CpuStorageBinding if found, nullptr otherwise
   */
  const CpuStorageBinding *findStorage(StorageId id) const {
    return storages_.find(id);
  }

  /**
   * @brief Find a storage binding by ID (mutable version).
   */
  CpuStorageBinding *findStorage(StorageId id) { return storages_.find(id); }

  /**
   * @brief Get the list of all storage bindings.
   */
  const auto &storageList() const { return storages_; }

  /**
   * @brief Get the storage list (mutable).
   */
  auto &storageList() { return storages_; }

  /**
   * @brief Get the number of storage bindings.
   */
  std::size_t storageCount() const { return storages_.size(); }

  /**
   * @brief Get the current storage capacity (may exceed inline capacity).
   */
  std::size_t storageCapacity() const { return storages_.capacity(); }

  /**
   * @brief Clear all storage bindings.
   */
  void clearStorages() { storages_.clear(); }

  /**
   * @brief Add a parameter.
   */
  void addParam(Param param) { params_.add(std::move(param)); }

  /**
   * @brief Find a parameter by ID.
   */
  const Param *findParam(ParamId id) const { return params_.find(id); }

  /**
   * @brief Find a parameter by ID (mutable version).
   */
  Param *findParam(ParamId id) { return params_.find(id); }

  /**
   * @brief Get the list of all parameters.
   */
  const auto &paramList() const { return params_; }

  /**
   * @brief Clear all parameters.
   */
  void clearParams() { params_.clear(); }

private:
  Context context_;
  StorageListType storages_{};
  ParamList params_{};
};

} // namespace orteaf::internal::kernel::cpu
