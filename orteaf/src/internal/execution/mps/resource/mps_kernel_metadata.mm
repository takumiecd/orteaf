#include "orteaf/internal/execution/mps/resource/mps_kernel_metadata.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace orteaf::internal::execution::mps::resource {

void MpsKernelMetadata::rebuild(
    ::orteaf::internal::kernel::core::KernelEntry &entry) const {
  // Acquire a new KernelBase lease with the stored keys and ExecuteFunc
  auto kernel_base_lease = ::orteaf::internal::execution::mps::api::
      MpsExecutionApi::acquireKernelBase(keys());

  // Set the ExecuteFunc on the acquired lease
  auto *base = kernel_base_lease.operator->();
  if (base) {
    base->setExecute(execute_);
  }

  entry.setBase(std::move(kernel_base_lease));
}

MpsKernelMetadata MpsKernelMetadata::buildFromBase(
    const ::orteaf::internal::execution::mps::resource::MpsKernelBase &base) {
  MpsKernelMetadata meta;
  meta.initialize(base.keys());
  meta.setExecute(base.execute());
  return meta;
}

::orteaf::internal::kernel::core::KernelMetadataLease
MpsKernelMetadata::buildMetadataLeaseFromBase(
    const ::orteaf::internal::execution::mps::resource::MpsKernelBase &base) {
  auto meta = buildFromBase(base);
  auto metadata_lease = ::orteaf::internal::execution::mps::api::
      MpsExecutionApi::acquireKernelMetadata(meta.keys());
  if (auto *meta_ptr = metadata_lease.operator->()) {
    meta_ptr->setExecute(meta.execute());
  }
  return ::orteaf::internal::kernel::core::KernelMetadataLease{
      std::move(metadata_lease)};
}

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
