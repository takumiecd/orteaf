#pragma once

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/manager/cuda_context_manager.h"
#include "orteaf/internal/execution/cuda/manager/cuda_device_manager.h"
#include "orteaf/internal/execution/cuda/manager/cuda_stream_manager.h"

namespace orteaf::internal::execution_context::cuda {

class Context {
public:
  using DeviceLease =
      ::orteaf::internal::execution::cuda::manager::CudaDeviceManager::DeviceLease;
  using ContextLease = ::orteaf::internal::execution::cuda::manager::
      CudaContextManager::ContextLease;
  using StreamLease =
      ::orteaf::internal::execution::cuda::manager::CudaStreamManager::StreamLease;

  /// @brief Create an empty context with no resources.
  Context() = default;

  /// @brief Create a context for the specified device with primary context and new stream.
  /// @param device The device handle to create the context for.
  explicit Context(::orteaf::internal::execution::cuda::CudaDeviceHandle device);

  /// @brief Create a context for the specified device and stream.
  /// @param device The device handle to create the context for.
  /// @param stream The stream handle to acquire.
  Context(::orteaf::internal::execution::cuda::CudaDeviceHandle device,
          ::orteaf::internal::execution::cuda::CudaStreamHandle stream);

  DeviceLease device{};
  ContextLease context{};
  StreamLease stream{};
};

} // namespace orteaf::internal::execution_context::cuda

#endif // ORTEAF_ENABLE_CUDA
