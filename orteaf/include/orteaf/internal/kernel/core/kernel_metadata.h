#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "orteaf/internal/execution/cpu/manager/cpu_kernel_metadata_manager.h"
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/manager/mps_kernel_metadata_manager.h"
#endif
#include "orteaf/internal/kernel/core/kernel_entry.h"

namespace orteaf::internal::kernel::core {

class KernelMetadataLease;

/**
 * @brief Type-erased kernel metadata lease.
 *
 * Holds backend-specific metadata that can rebuild a KernelEntry.
 */
class KernelMetadataLease {
public:
  using CpuKernelMetadataLease = ::orteaf::internal::execution::cpu::manager::
      CpuKernelMetadataManager::CpuKernelMetadataLease;

#if ORTEAF_ENABLE_MPS
  using MpsKernelMetadataLease = ::orteaf::internal::execution::mps::manager::
      MpsKernelMetadataManager::MpsKernelMetadataLease;
#endif

  using Variant = std::variant<std::monostate, CpuKernelMetadataLease
#if ORTEAF_ENABLE_MPS
                               ,
                               MpsKernelMetadataLease
#endif
                               >;

  KernelMetadataLease() = default;

  explicit KernelMetadataLease(Variant lease) noexcept
      : lease_(std::move(lease)) {}

  Variant &lease() noexcept { return lease_; }
  const Variant &lease() const noexcept { return lease_; }

  void setLease(Variant lease) noexcept { lease_ = std::move(lease); }

  /**
   * @brief Rebuild a KernelEntry from this metadata.
   *
   * Delegates to the backend-specific Metadata::rebuild() method.
   */
  ::orteaf::internal::kernel::core::KernelEntry rebuild() const;

  /**
   * @brief Build metadata from a KernelEntry.
   *
   * Delegates to backend-specific Metadata::buildMetadataLeaseFromBase()
   * methods for lease acquisition.
   */
  static KernelMetadataLease
  fromEntry(const ::orteaf::internal::kernel::core::KernelEntry &entry);

private:
  Variant lease_{};
};

} // namespace orteaf::internal::kernel::core
