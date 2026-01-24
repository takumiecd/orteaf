#pragma once

#include <orteaf/internal/kernel/storage_binding.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::kernel::cpu {

/**
 * @brief CPU storage binding type alias.
 *
 * Represents a bound CPU storage resource with its identifier and lease.
 * Access pattern information is available through the StorageId metadata.
 *
 * This is a specialization of the generic StorageBinding template for CPU execution.
 */
using CpuStorageBinding = ::orteaf::internal::kernel::StorageBinding<
    ::orteaf::internal::storage::CpuStorageLease>;

} // namespace orteaf::internal::kernel::cpu
