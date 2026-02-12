#include "orteaf/internal/execution/cuda/resource/cuda_kernel_metadata.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace orteaf::internal::execution::cuda::resource {

void CudaKernelMetadata::rebuild(
    ::orteaf::internal::kernel::core::KernelEntry &entry) const {
  auto kernel_base_lease =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi::
          acquireKernelBase(keys());

  auto *base = kernel_base_lease.operator->();
  if (base != nullptr) {
    base->setExecute(execute_);
  }

  entry.setBase(std::move(kernel_base_lease));
}

CudaKernelMetadata CudaKernelMetadata::buildFromBase(
    const ::orteaf::internal::execution::cuda::resource::CudaKernelBase &base) {
  CudaKernelMetadata metadata;
  metadata.initialize(base.keys());
  metadata.setExecute(base.execute());
  return metadata;
}

::orteaf::internal::kernel::core::KernelMetadataLease
CudaKernelMetadata::buildMetadataLeaseFromBase(
    const ::orteaf::internal::execution::cuda::resource::CudaKernelBase &base) {
  auto metadata = buildFromBase(base);
  auto metadata_lease =
      ::orteaf::internal::execution::cuda::api::CudaExecutionApi::
          acquireKernelMetadata(metadata.keys());
  if (auto *metadata_ptr = metadata_lease.operator->()) {
    metadata_ptr->setExecute(metadata.execute());
  }
  return ::orteaf::internal::kernel::core::KernelMetadataLease{
      std::move(metadata_lease)};
}

} // namespace orteaf::internal::execution::cuda::resource

#endif // ORTEAF_ENABLE_CUDA
