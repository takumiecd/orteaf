#pragma once

#include "orteaf/internal/execution/cpu/resource/cpu_kernel_base.h"

namespace orteaf::internal::kernel::core {
class KernelEntry;
class KernelMetadataLease;
} // namespace orteaf::internal::kernel::core

namespace orteaf::internal::execution::cpu::resource {

/**
 * @brief Kernel metadata resource for CPU.
 *
 * Stores ExecuteFunc for kernel reconstruction.
 */
struct CpuKernelMetadata {
  using ExecuteFunc =
      ::orteaf::internal::execution::cpu::resource::CpuKernelBase::ExecuteFunc;

  CpuKernelMetadata() = default;

  ExecuteFunc execute() const noexcept { return execute_; }
  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  // rebuild KernelEntry from Metadata
  void rebuild(::orteaf::internal::kernel::core::KernelEntry &entry) const;

  // Build Metadata from KernelBase
  static CpuKernelMetadata buildFromBase(const CpuKernelBase &base);

  // Build a metadata lease from KernelBase
  static ::orteaf::internal::kernel::core::KernelMetadataLease
  buildMetadataLeaseFromBase(const CpuKernelBase &base);

private:
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::execution::cpu::resource
