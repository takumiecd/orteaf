#pragma once

#if ORTEAF_ENABLE_MPS

#include <orteaf/internal/kernel/storage_id.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::mps {

/**
 * @brief MPS storage binding structure for kernel arguments.
 *
 * Represents a bound MPS storage resource with its identifier and lease.
 * Access pattern information is available through the StorageId metadata.
 */
struct MpsStorageBinding {
  /**
   * @brief Storage identifier.
   *
   * Identifies the semantic role of this storage (e.g., Input0, Output).
   * Access pattern can be queried via StorageTypeInfo<id>::kAccess.
   */
  StorageId id;

  /**
   * @brief MPS storage lease.
   *
   * Holds the actual MPS storage resource.
   */
  ::orteaf::internal::storage::MpsStorageLease lease;
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
