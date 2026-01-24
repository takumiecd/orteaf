#pragma once

#if ORTEAF_ENABLE_MPS

#include <orteaf/internal/kernel/storage_binding.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::mps {

/**
 * @brief MPS storage binding type alias.
 *
 * Represents a bound MPS storage resource with its identifier and lease.
 * Access pattern information is available through the StorageId metadata.
 *
 * This is a specialization of the generic StorageBinding template for MPS execution.
 */
using MpsStorageBinding = ::orteaf::internal::kernel::StorageBinding<
    ::orteaf::internal::storage::MpsStorageLease>;

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
