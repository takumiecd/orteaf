#pragma once

/**
 * @file storage_types.h
 * @brief Registration of Storage types.
 *
 * This file is where contributors register new Storage types.
 * Managers are auto-generated via TypedStorageManager<Storage>.
 *
 * Adding to RegisteredStorages is all you need - everything else is automatic.
 */

#include <orteaf/internal/storage/cpu/cpu_storage.h>
#include <orteaf/internal/storage/registry/storage_registry.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/storage/mps/mps_storage.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/storage/cuda/cuda_storage.h>
#endif

namespace orteaf::internal::storage::registry {

// =============================================================================
// StorageTraits Specializations
// =============================================================================

template <> struct StorageTraits<::orteaf::internal::storage::cpu::CpuStorage> {
  using Manager = manager::TypedStorageManager<
      ::orteaf::internal::storage::cpu::CpuStorage>;
  using Lease = typename Manager::StorageLease;
  static constexpr const char *name = "cpu";
};

#if ORTEAF_ENABLE_MPS
template <> struct StorageTraits<::orteaf::internal::storage::mps::MpsStorage> {
  using Manager = manager::TypedStorageManager<
      ::orteaf::internal::storage::mps::MpsStorage>;
  using Lease = typename Manager::StorageLease;
  static constexpr const char *name = "mps";
};
#endif

#if ORTEAF_ENABLE_CUDA
template <>
struct StorageTraits<::orteaf::internal::storage::cuda::CudaStorage> {
  using Manager = manager::TypedStorageManager<
      ::orteaf::internal::storage::cuda::CudaStorage>;
  using Lease = typename Manager::StorageLease;
  static constexpr const char *name = "cuda";
};
#endif

// =============================================================================
// Registered Storage Types
// =============================================================================

using RegisteredStorages =
    StorageRegistry<::orteaf::internal::storage::cpu::CpuStorage
#if ORTEAF_ENABLE_MPS
                    ,
                    ::orteaf::internal::storage::mps::MpsStorage
#endif
#if ORTEAF_ENABLE_CUDA
                    ,
                    ::orteaf::internal::storage::cuda::CudaStorage
#endif
                    >;

// =============================================================================
// Type Aliases for convenience
// =============================================================================

using CpuStorageManager =
    StorageTraits<::orteaf::internal::storage::cpu::CpuStorage>::Manager;
using CpuStorageLease =
    StorageTraits<::orteaf::internal::storage::cpu::CpuStorage>::Lease;

#if ORTEAF_ENABLE_MPS
using MpsStorageManager =
    StorageTraits<::orteaf::internal::storage::mps::MpsStorage>::Manager;
using MpsStorageLease =
    StorageTraits<::orteaf::internal::storage::mps::MpsStorage>::Lease;
#endif

#if ORTEAF_ENABLE_CUDA
using CudaStorageManager =
    StorageTraits<::orteaf::internal::storage::cuda::CudaStorage>::Manager;
using CudaStorageLease =
    StorageTraits<::orteaf::internal::storage::cuda::CudaStorage>::Lease;
#endif

} // namespace orteaf::internal::storage::registry

// Re-export for convenience
namespace orteaf::internal::storage {
using RegisteredStorages = registry::RegisteredStorages;
using CpuStorageManager = registry::CpuStorageManager;
using CpuStorageLease = registry::CpuStorageLease;
#if ORTEAF_ENABLE_MPS
using MpsStorageManager = registry::MpsStorageManager;
using MpsStorageLease = registry::MpsStorageLease;
#endif
#if ORTEAF_ENABLE_CUDA
using CudaStorageManager = registry::CudaStorageManager;
using CudaStorageLease = registry::CudaStorageLease;
#endif
} // namespace orteaf::internal::storage
