#pragma once

#include <orteaf/internal/kernel/storage/operand_key.h>
#include <orteaf/internal/storage/storage_lease.h>

namespace orteaf::internal::kernel {

/**
 * @brief Storage binding structure for kernel arguments.
 *
 * Represents a bound storage resource with its operand key and lease.
 * Uses the type-erased StorageLease to avoid backend-specific bindings.
 * Access pattern information is available through the OperandId metadata.
 *
 * Example:
 * @code
 * StorageBinding binding{makeOperandKey(OperandId::Input0), lease};
 * @endcode
 */
struct StorageBinding {
  /**
   * @brief Operand key.
   *
   * Identifies the semantic role of this storage (e.g., Input0, Output) and
   * the tensor-internal role (e.g., Data, Index).
   * Access pattern can be queried via OperandTypeInfo<id>::kAccess.
   */
  OperandKey key;

  /**
   * @brief Backend-specific storage lease.
   *
   * Holds the actual storage resource for the target execution backend.
   */
  ::orteaf::internal::storage::StorageLease lease;
};

} // namespace orteaf::internal::kernel
