#pragma once

#include <orteaf/internal/kernel/storage/storage_id.h>

namespace orteaf::internal::kernel {

/**
 * @brief Generic storage binding structure for kernel arguments.
 *
 * Represents a bound storage resource with its identifier and lease.
 * This template is specialized for different execution backends (CPU, MPS, CUDA).
 * Access pattern information is available through the StorageId metadata.
 *
 * @tparam StorageLease The backend-specific storage lease type
 *                      (e.g., CpuStorageLease, MpsStorageLease, CudaStorageLease)
 *
 * Example:
 * @code
 * using CpuStorageBinding = StorageBinding<CpuStorageLease>;
 * using MpsStorageBinding = StorageBinding<MpsStorageLease>;
 * @endcode
 */
template <typename StorageLease>
struct StorageBinding {
  /**
   * @brief Storage identifier.
   *
   * Identifies the semantic role of this storage (e.g., Input0, Output).
   * Access pattern can be queried via StorageTypeInfo<id>::kAccess.
   */
  StorageId id;

  /**
   * @brief Backend-specific storage lease.
   *
   * Holds the actual storage resource for the target execution backend.
   */
  StorageLease lease;
};

} // namespace orteaf::internal::kernel
