#pragma once

#include <utility>

#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/context_any.h>
#include <orteaf/internal/kernel/param/param_list.h>
#include <orteaf/internal/kernel/storage/storage_binding.h>
#include <orteaf/internal/kernel/storage/operand_key.h>
#include <orteaf/internal/kernel/storage/storage_list.h>
#include <orteaf/internal/storage/storage_lease.h>

namespace orteaf::internal::kernel {

/**
 * @brief Type-erased kernel arguments container.
 *
 * Holds execution context, storage bindings, and parameters without
 * backend-specific subclasses.
 */
class KernelArgs {
public:
  using Execution = ::orteaf::internal::execution::Execution;
  using Architecture = ::orteaf::internal::architecture::Architecture;
  using Context = ContextAny;
  using StorageLease = ::orteaf::internal::storage::StorageLease;
  using StorageBindingType = StorageBinding;
  using StorageListType = StorageList<StorageBindingType>;

  KernelArgs() = default;

  explicit KernelArgs(Context context) : context_(std::move(context)) {}

  /**
   * @brief Get the execution context.
   */
  const Context &context() const { return context_; }
  Context &context() { return context_; }

  /**
   * @brief Check if the KernelArgs has a valid context.
   */
  bool valid() const { return context_.valid(); }

  /**
   * @brief Return the execution backend for this KernelArgs.
   */
  Execution execution() const { return context_.execution(); }

  /**
   * @brief Return the current architecture for this KernelArgs.
   */
  Architecture architecture() const { return context_.architecture(); }

  /**
   * @brief Add a storage binding with the specified operand key.
   */
  void addStorage(OperandKey key, StorageLease lease) {
    storages_.add(StorageBindingType{key, std::move(lease)});
  }

  /**
   * @brief Add a storage binding with the specified ID (Data role).
   */
  void addStorage(OperandId id, StorageLease lease) {
    addStorage(makeOperandKey(id), std::move(lease));
  }

  /**
   * @brief Find a storage binding by operand key.
   */
  const StorageBindingType *findStorage(OperandKey key) const {
    return storages_.find(key);
  }

  /**
   * @brief Find a storage binding by operand (mutable).
   */
  StorageBindingType *findStorage(OperandKey key) { return storages_.find(key); }

  /**
   * @brief Find a storage binding by ID (Data role).
   */
  const StorageBindingType *findStorage(OperandId id) const {
    return storages_.find(id);
  }

  /**
   * @brief Find a storage binding by ID (mutable, Data role).
   */
  StorageBindingType *findStorage(OperandId id) { return storages_.find(id); }

  /**
   * @brief Get the list of all storage bindings.
   */
  const auto &storageList() const { return storages_; }
  auto &storageList() { return storages_; }

  /**
   * @brief Get the number of storage bindings.
   */
  std::size_t storageCount() const { return storages_.size(); }

  /**
   * @brief Get the current storage capacity.
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
   * @brief Find a parameter by ID (global).
   */
  const Param *findParam(ParamId id) const { return params_.find(id); }

  /**
   * @brief Find a parameter by key.
   */
  const Param *findParam(ParamKey key) const { return params_.find(key); }

  /**
   * @brief Find a parameter by ID (mutable, global).
   */
  Param *findParam(ParamId id) { return params_.find(id); }

  /**
   * @brief Find a parameter by key (mutable).
   */
  Param *findParam(ParamKey key) { return params_.find(key); }

  /**
   * @brief Get the list of all parameters.
   */
  const auto &paramList() const { return params_; }

  /**
   * @brief Clear all parameters.
   */
  void clearParams() { params_.clear(); }

private:
  Context context_{};
  StorageListType storages_{};
  ParamList params_{};
};

} // namespace orteaf::internal::kernel
