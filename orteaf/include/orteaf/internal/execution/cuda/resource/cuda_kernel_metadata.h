#pragma once

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"

namespace orteaf::internal::kernel::core {
class KernelMetadataLease;
class KernelEntry;
}

namespace orteaf::internal::execution::cuda::resource {

struct CudaKernelMetadata {
  using ExecuteFunc =
      ::orteaf::internal::execution::cuda::resource::CudaKernelBase::ExecuteFunc;
  using Key = ::orteaf::internal::execution::cuda::resource::CudaKernelBase::Key;

  bool initialize(const ::orteaf::internal::base::HeapVector<Key> &keys) {
    reset();
    keys_.reserve(keys.size());
    for (const auto &key : keys) {
      keys_.pushBack(key);
    }
    return true;
  }

  void reset() noexcept { keys_.clear(); }

  const ::orteaf::internal::base::HeapVector<Key> &keys() const noexcept {
    return keys_;
  }

  ExecuteFunc execute() const noexcept { return execute_; }
  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  void rebuild(::orteaf::internal::kernel::core::KernelEntry &entry) const;

  static CudaKernelMetadata buildFromBase(
      const ::orteaf::internal::execution::cuda::resource::CudaKernelBase
          &base);

  static ::orteaf::internal::kernel::core::KernelMetadataLease
  buildMetadataLeaseFromBase(
      const ::orteaf::internal::execution::cuda::resource::CudaKernelBase
          &base);

private:
  ::orteaf::internal::base::HeapVector<Key> keys_{};
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::execution::cuda::resource

#endif // ORTEAF_ENABLE_CUDA
