#pragma once

#include <orteaf/internal/kernel/storage_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU storage binding structure for kernel arguments.
 *
 * Represents a bound CPU storage resource with its identifier and lease.
 * Access pattern information is available through the StorageId metadata.
 */
struct CpuStorageBinding {
  /**
   * @brief Storage identifier.
   *
   * Identifies the semantic role of this storage (e.g., Input0, Output).
   * Access pattern can be queried via StorageTypeInfo<id>::kAccess.
   */
  StorageId id;

  /**
   * @brief CPU storage lease.
   *
   * Holds the actual CPU storage resource.
   */
  ::orteaf::internal::storage::CpuStorageLease lease;
};

} // namespace orteaf::internal::kernel::cpu
