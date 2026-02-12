#include "orteaf/internal/kernel/core/kernel_metadata.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#endif
#if ORTEAF_ENABLE_CUDA
#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#endif

namespace orteaf::internal::kernel::core {

namespace detail {

KernelEntry::KernelBaseLease rebuildCpuBase(
    const KernelMetadataLease::CpuKernelMetadataLease &metadata_lease) {
  if (!metadata_lease) {
    return std::monostate{};
  }
  auto *metadata = metadata_lease.operator->();
  if (metadata == nullptr) {
    return std::monostate{};
  }

  auto base_lease =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi::
          executionManager()
              .kernelBaseManager()
              .acquire(*metadata);
  if (!base_lease) {
    return std::monostate{};
  }
  return KernelEntry::KernelBaseLease{std::move(base_lease)};
}

#if ORTEAF_ENABLE_MPS
KernelEntry::KernelBaseLease rebuildMpsBase(
    const KernelMetadataLease::MpsKernelMetadataLease &metadata_lease) {
  if (!metadata_lease) {
    return std::monostate{};
  }
  auto *metadata = metadata_lease.operator->();
  if (metadata == nullptr) {
    return std::monostate{};
  }

  auto base_lease =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi::
          executionManager()
              .kernelBaseManager()
              .acquire(*metadata);
  if (!base_lease) {
    return std::monostate{};
  }
  return KernelEntry::KernelBaseLease{std::move(base_lease)};
}
#endif

#if ORTEAF_ENABLE_CUDA
KernelEntry::KernelBaseLease rebuildCudaBase(
    const KernelMetadataLease::CudaKernelMetadataLease &metadata_lease) {
  if (!metadata_lease) {
    return std::monostate{};
  }
  auto *metadata = metadata_lease.operator->();
  if (metadata == nullptr) {
    return std::monostate{};
  }

  auto base_lease =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi::
          executionManager()
              .kernelBaseManager()
              .acquire(*metadata);
  if (!base_lease) {
    return std::monostate{};
  }
  return KernelEntry::KernelBaseLease{std::move(base_lease)};
}
#endif

KernelMetadataLease::Variant
buildCpuMetadata(const KernelEntry::CpuKernelBaseLease &base_lease) {
  if (!base_lease) {
    return std::monostate{};
  }
  auto *base = base_lease.operator->();
  if (base == nullptr) {
    return std::monostate{};
  }

  ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata metadata{};
  metadata.setExecute(base->execute());
  auto metadata_lease =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi::
          executionManager()
              .kernelMetadataManager()
              .acquire(metadata);
  if (!metadata_lease) {
    return std::monostate{};
  }
  return KernelMetadataLease::Variant{std::move(metadata_lease)};
}

#if ORTEAF_ENABLE_MPS
KernelMetadataLease::Variant
buildMpsMetadata(const KernelEntry::MpsKernelBaseLease &base_lease) {
  if (!base_lease) {
    return std::monostate{};
  }
  auto *base = base_lease.operator->();
  if (base == nullptr) {
    return std::monostate{};
  }

  ::orteaf::internal::execution::mps::resource::MpsKernelMetadata metadata{};
  metadata.initialize(base->keys());
  metadata.setExecute(base->execute());
  auto metadata_lease =
      ::orteaf::internal::execution::mps::api::MpsExecutionApi::
          executionManager()
              .kernelMetadataManager()
              .acquire(metadata);
  if (!metadata_lease) {
    return std::monostate{};
  }
  return KernelMetadataLease::Variant{std::move(metadata_lease)};
}
#endif

#if ORTEAF_ENABLE_CUDA
KernelMetadataLease::Variant
buildCudaMetadata(const KernelEntry::CudaKernelBaseLease &base_lease) {
  if (!base_lease) {
    return std::monostate{};
  }
  auto *base = base_lease.operator->();
  if (base == nullptr) {
    return std::monostate{};
  }

  ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata metadata{};
  metadata.initialize(base->keys());
  metadata.setExecute(base->execute());
  auto metadata_lease =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi::
          executionManager()
              .kernelMetadataManager()
              .acquire(metadata);
  if (!metadata_lease) {
    return std::monostate{};
  }
  return KernelMetadataLease::Variant{std::move(metadata_lease)};
}
#endif

} // namespace detail

KernelEntry KernelMetadataLease::rebuild() const {
  KernelEntry entry{};
  std::visit(
      [&](const auto &metadata_lease) {
        using T = std::decay_t<decltype(metadata_lease)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return;
        } else if constexpr (std::is_same_v<T, CpuKernelMetadataLease>) {
          entry.setBase(detail::rebuildCpuBase(metadata_lease));
#if ORTEAF_ENABLE_MPS
        } else if constexpr (std::is_same_v<T, MpsKernelMetadataLease>) {
          entry.setBase(detail::rebuildMpsBase(metadata_lease));
#endif
#if ORTEAF_ENABLE_CUDA
        } else if constexpr (std::is_same_v<T, CudaKernelMetadataLease>) {
          entry.setBase(detail::rebuildCudaBase(metadata_lease));
#endif
        }
      },
      lease_);
  return entry;
}

KernelMetadataLease
KernelMetadataLease::fromEntry(const KernelEntry &entry) {
  KernelMetadataLease metadata{};
  std::visit(
      [&](const auto &base_lease) {
        using T = std::decay_t<decltype(base_lease)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          metadata.setLease(KernelMetadataLease::Variant{});
        } else if constexpr (std::is_same_v<T, KernelEntry::CpuKernelBaseLease>) {
          metadata.setLease(detail::buildCpuMetadata(base_lease));
#if ORTEAF_ENABLE_MPS
        } else if constexpr (std::is_same_v<T,
                                            KernelEntry::MpsKernelBaseLease>) {
          metadata.setLease(detail::buildMpsMetadata(base_lease));
#endif
#if ORTEAF_ENABLE_CUDA
        } else if constexpr (std::is_same_v<T,
                                            KernelEntry::CudaKernelBaseLease>) {
          metadata.setLease(detail::buildCudaMetadata(base_lease));
#endif
        }
      },
      entry.base());
  return metadata;
}

} // namespace orteaf::internal::kernel::core
