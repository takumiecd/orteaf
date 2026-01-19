#include "orteaf/internal/execution_context/cuda/context.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"

namespace orteaf::internal::execution_context::cuda {

Context::Context(::orteaf::internal::execution::cuda::CudaDeviceHandle device) {
  namespace cuda_api = ::orteaf::internal::execution::cuda::api;

  this->device = cuda_api::CudaExecutionApi::acquireDevice(device);
  if (auto *device_resource = this->device.operator->()) {
    this->context = device_resource->context_manager.acquirePrimary();
    if (auto *context_resource = this->context.operator->()) {
      this->stream = context_resource->stream_manager.acquire();
    }
  }
}

Context::Context(::orteaf::internal::execution::cuda::CudaDeviceHandle device,
                 ::orteaf::internal::execution::cuda::CudaStreamHandle stream) {
  namespace cuda_api = ::orteaf::internal::execution::cuda::api;

  this->device = cuda_api::CudaExecutionApi::acquireDevice(device);
  if (auto *device_resource = this->device.operator->()) {
    this->context = device_resource->context_manager.acquirePrimary();
    if (auto *context_resource = this->context.operator->()) {
      this->stream = context_resource->stream_manager.acquire(stream);
    }
  }
}

} // namespace orteaf::internal::execution_context::cuda

#endif // ORTEAF_ENABLE_CUDA
