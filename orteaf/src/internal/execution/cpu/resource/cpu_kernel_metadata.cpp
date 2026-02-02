#include "orteaf/internal/execution/cpu/resource/cpu_kernel_metadata.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace orteaf::internal::execution::cpu::resource {

void CpuKernelMetadata::rebuild(
    ::orteaf::internal::kernel::core::KernelEntry &entry) const {
  // Acquire a new KernelBase lease with the stored ExecuteFunc
  auto kernel_base_lease = ::orteaf::internal::execution::cpu::api::
      CpuExecutionApi::acquireKernelBase(execute_);
  entry.setBase(std::move(kernel_base_lease));
}

CpuKernelMetadata CpuKernelMetadata::buildFromBase(const CpuKernelBase &base) {
  CpuKernelMetadata meta;
  meta.setExecute(base.execute());
  return meta;
}

::orteaf::internal::kernel::core::KernelMetadataLease
CpuKernelMetadata::buildMetadataLeaseFromBase(const CpuKernelBase &base) {
  auto metadata_lease =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi::
          acquireKernelMetadata(base.execute());
  return ::orteaf::internal::kernel::core::KernelMetadataLease{
      std::move(metadata_lease)};
}

} // namespace orteaf::internal::execution::cpu::resource
