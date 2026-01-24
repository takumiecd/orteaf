#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstring>
#include <utility>

#include <orteaf/internal/base/small_vector.h>
#include <orteaf/internal/kernel/param.h>

#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/access.h>
#include <orteaf/internal/kernel/kernel_key.h>
#include <orteaf/internal/kernel/mps/mps_storage_binding.h>
#include <orteaf/internal/kernel/storage_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::mps {

/**
 * @brief MPS kernel arguments container.
 *
 * Manages storage bindings and parameters for MPS kernel execution.
 * Uses structured storage bindings with StorageId for type-safe management.
 * Holds execution context for device and resource management.
 */
class MpsKernelArgs {
public:
  using Context = ::orteaf::internal::execution_context::mps::Context;
  using StorageLease = ::orteaf::internal::storage::MpsStorageLease;

  // Inline capacities; SmallVector can grow beyond these values.
  static constexpr std::size_t kMaxBindings = 16;
  static constexpr std::size_t kMaxParams = 16;

  /**
   * @brief Create kernel args from current execution context.
   */
  static MpsKernelArgs fromCurrentContext();

  /**
   * @brief Default constructor (uses current context).
   */
  MpsKernelArgs();

  /**
   * @brief Construct with explicit context.
   */
  explicit MpsKernelArgs(Context context);

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
    storages_.pushBack(MpsStorageBinding{id, std::move(lease)});
  }

  /**
   * @brief Find a storage binding by ID.
   *
   * @param id Storage identifier to search for
   * @return Pointer to MpsStorageBinding if found, nullptr otherwise
   */
  const MpsStorageBinding *findStorage(StorageId id) const {
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
  MpsStorageBinding *findStorage(StorageId id) {
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
  ::orteaf::internal::base::SmallVector<MpsStorageBinding, kMaxBindings>
      storages_{};
  ParamList params_{};
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
