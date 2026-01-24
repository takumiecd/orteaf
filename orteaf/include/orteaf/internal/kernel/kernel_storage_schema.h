#pragma once

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/kernel/storage_id.h>
#include <orteaf/internal/kernel/storage_list.h>
#include <orteaf/kernel/storage_id_tables.h>

#include <utility>

namespace orteaf::internal::kernel {

/**
 * @brief Storage field type for kernel storage schema.
 *
 * Associates a StorageId with its binding and provides automatic extraction
 * from KernelArgs. Provides access to the storage lease.
 *
 * @tparam ID Storage identifier
 *
 * Example:
 * @code
 * StorageField<StorageId::Input0> input;
 * input.extract(args);
 * auto& lease = input.lease();
 * @endcode
 */
template <StorageId ID> struct StorageField {
  static constexpr StorageId kId = ID;

  /**
   * @brief Check if storage binding was found.
   */
  explicit operator bool() const { return binding_ != nullptr; }

  /**
   * @brief Get the storage binding.
   *
   * @throws std::runtime_error if binding not found
   */
  template <typename StorageBinding> const StorageBinding &binding() const {
    if (!binding_) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Required storage binding not found");
    }
    return *static_cast<const StorageBinding *>(binding_);
  }

  /**
   * @brief Get the storage lease.
   *
   * @tparam StorageBinding The storage binding type (MpsStorageBinding,
   * CpuStorageBinding)
   * @throws std::runtime_error if binding not found
   */
  template <typename StorageBinding> auto &lease() const {
    return binding<StorageBinding>().lease;
  }

  /**
   * @brief Get the storage lease (mutable).
   *
   * @tparam StorageBinding The storage binding type
   * @throws std::runtime_error if binding not found
   */
  template <typename StorageBinding> auto &lease() {
    if (!binding_) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Required storage binding not found");
    }
    return const_cast<StorageBinding *>(
               static_cast<const StorageBinding *>(binding_))
        ->lease;
  }

  /**
   * @brief Get the access pattern for this storage.
   */
  static constexpr ::orteaf::internal::kernel::Access access() {
    return ::orteaf::generated::storage_id_tables::StorageTypeInfo<
        kId>::kAccess;
  }

  /**
   * @brief Extract storage binding from storage list.
   *
   * @tparam StorageBinding The storage binding type
   * @param storages Storage list to extract from
   * @throws std::runtime_error if storage not found
   */
  template <typename StorageBinding>
  void extract(const StorageList<StorageBinding> &storages) {
    binding_ = storages.find(kId);
    if (!binding_) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "Required storage binding not found");
    }
  }

  /**
   * @brief Extract storage binding from kernel arguments.
   *
   * @tparam KernelArgs The kernel arguments type (CpuKernelArgs, MpsKernelArgs,
   * etc.)
   * @param args Kernel arguments containing storage bindings
   * @throws std::runtime_error if storage not found
   */
  template <typename KernelArgs> void extract(const KernelArgs &args) {
    using StorageBinding =
        typename KernelArgs::StorageListType::Storage::value_type;
    extract<StorageBinding>(args.storageList());
  }

private:
  const void *binding_ = nullptr;
};

/**
 * @brief Optional storage field type for kernel storage schema.
 *
 * Similar to StorageField but allows missing storage bindings.
 *
 * @tparam ID Storage identifier
 */
template <StorageId ID> struct OptionalStorageField {
  static constexpr StorageId kId = ID;

  /**
   * @brief Check if storage binding was found.
   */
  explicit operator bool() const { return binding_ != nullptr; }

  /**
   * @brief Check if storage is present.
   */
  bool present() const { return binding_ != nullptr; }

  /**
   * @brief Get the storage binding if present.
   *
   * @tparam StorageBinding The storage binding type
   * @return Pointer to binding, or nullptr if not present
   */
  template <typename StorageBinding>
  const StorageBinding *
  bindingOr(const StorageBinding *defaultValue = nullptr) const {
    if (!binding_) {
      return defaultValue;
    }
    return static_cast<const StorageBinding *>(binding_);
  }

  /**
   * @brief Get the storage lease if present.
   *
   * @tparam StorageBinding The storage binding type
   * @return Pointer to lease, or nullptr if not present
   */
  template <typename StorageBinding>
  auto *leaseOr(decltype(nullptr) = nullptr) const {
    if (!binding_) {
      return static_cast<decltype(&static_cast<const StorageBinding *>(nullptr)
                                       ->lease)>(nullptr);
    }
    return &static_cast<const StorageBinding *>(binding_)->lease;
  }

  /**
   * @brief Get the access pattern for this storage.
   */
  static constexpr ::orteaf::internal::kernel::Access access() {
    return ::orteaf::generated::storage_id_tables::StorageTypeInfo<
        kId>::kAccess;
  }

  /**
   * @brief Extract storage binding from storage list (optional).
   *
   * Does not throw if storage not found.
   *
   * @tparam StorageBinding The storage binding type
   * @param storages Storage list to extract from
   */
  template <typename StorageBinding>
  void extract(const StorageList<StorageBinding> &storages) {
    binding_ = storages.find(kId);
  }

  /**
   * @brief Extract storage binding from kernel arguments (optional).
   *
   * Does not throw if storage not found.
   *
   * @tparam KernelArgs The kernel arguments type
   * @param args Kernel arguments containing storage bindings
   */
  template <typename KernelArgs> void extract(const KernelArgs &args) {
    using StorageBinding =
        typename KernelArgs::StorageListType::Storage::value_type;
    extract<StorageBinding>(args.storageList());
  }

private:
  const void *binding_ = nullptr;
};

/**
 * @brief Base class for storage schemas using CRTP.
 *
 * Provides static extract() method that calls extractAllStorages()
 * on the derived schema class.
 *
 * @tparam Derived The derived schema class
 *
 * Example:
 * @code
 * struct MyStorages : StorageSchema<MyStorages> {
 *   StorageField<StorageId::Input0> input;
 *   StorageField<StorageId::Output0> output;
 *
 *   ORTEAF_EXTRACT_STORAGES(input, output)
 * };
 *
 * auto storages = MyStorages::extract(args);
 * @endcode
 */
template <typename Derived> struct StorageSchema {
  /**
   * @brief Extract all storage fields from kernel arguments.
   *
   * Creates a new schema instance and calls extractAllStorages()
   * on it to populate all field bindings.
   *
   * @tparam KernelArgs The kernel arguments type
   * @param args Kernel arguments containing storage bindings
   * @return Populated schema instance
   */
  template <typename KernelArgs>
  static Derived extract(const KernelArgs &args) {
    Derived schema;
    schema.extractAllStorages(args);
    return schema;
  }
};

namespace detail {
// Helper for extracting multiple storage fields from KernelArgs
template <typename KernelArgs, typename... Fields>
void extractStorages(const KernelArgs &args, Fields &...fields) {
  (fields.extract(args), ...);
}

} // namespace detail

} // namespace orteaf::internal::kernel

/**
 * @brief Macro to generate extractAllStorages().
 *
 * Automatically generates the extraction logic for all listed storage fields.
 *
 * Note: For binding storages to encoder, use MpsKernelBase::bindStoragesAt()
 * with explicit indices to ensure type safety with Metal shader bindings.
 *
 * Usage:
 * @code
 * struct MyStorages : StorageSchema<MyStorages> {
 *   StorageField<StorageId::Input0> input;
 *   StorageField<StorageId::Output> output;
 *   OptionalStorageField<StorageId::Workspace> workspace;
 *
 *   ORTEAF_EXTRACT_STORAGES(input, output, workspace)
 * };
 *
 * auto storages = MyStorages::extract(args);
 * base.bindStoragesAt(encoder, base.Indices<0, 1, 2>{},
 *                     storages.input, storages.output, storages.workspace);
 * @endcode
 */
#define ORTEAF_EXTRACT_STORAGES(...)                                           \
  template <typename KernelArgs>                                               \
  void extractAllStorages(const KernelArgs &args) {                            \
    ::orteaf::internal::kernel::detail::extractStorages(args, __VA_ARGS__);    \
  }
