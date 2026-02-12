#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <memory>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/manager/cuda_execution_manager.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"

namespace orteaf::internal::execution::cuda::api {

class CudaExecutionApi {
public:
  using ExecutionManager =
      ::orteaf::internal::execution::cuda::manager::CudaExecutionManager;
  using DeviceHandle = ::orteaf::internal::execution::cuda::CudaDeviceHandle;
  using ContextHandle = ::orteaf::internal::execution::cuda::CudaContextHandle;
  using StreamHandle = ::orteaf::internal::execution::cuda::CudaStreamHandle;
  using DeviceLease = ::orteaf::internal::execution::cuda::manager::
      CudaDeviceManager::DeviceLease;
  using ContextLease = ::orteaf::internal::execution::cuda::manager::
      CudaContextManager::ContextLease;
  using StreamLease = ::orteaf::internal::execution::cuda::manager::
      CudaStreamManager::StreamLease;
  using ModuleKey = ::orteaf::internal::execution::cuda::manager::ModuleKey;
  using KernelKey = ::orteaf::internal::execution::cuda::manager::
      CudaKernelBaseManager::Key;
  using KernelKeys = ::orteaf::internal::base::HeapVector<KernelKey>;
  using KernelBaseLease = ::orteaf::internal::execution::cuda::manager::
      CudaKernelBaseManager::KernelBaseLease;
  using KernelMetadataLease = ::orteaf::internal::execution::cuda::manager::
      CudaKernelMetadataManager::CudaKernelMetadataLease;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;

  CudaExecutionApi() = delete;

  // Configure execution manager with default configuration.
  static void configure() { manager().configure(); }

  // Configure execution manager with the provided configuration.
  static void configure(const ExecutionManager::Config &config) {
    manager().configure(config);
  }

  static void shutdown() { manager().shutdown(); }

  // Acquire a device lease for the given device handle.
  static DeviceLease acquireDevice(DeviceHandle device) {
    auto device_lease = manager().deviceManager().acquire(device);
    if (!device_lease.operator->()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA device lease has no payload");
    }
    return device_lease;
  }

  // Acquire a primary context lease for the given device.
  static ContextLease acquirePrimaryContext(DeviceHandle device) {
    auto device_lease = acquireDevice(device);
    auto *device_resource = device_lease.operator->();
    auto context_lease = device_resource->context_manager.acquirePrimary();
    if (!context_lease.operator->()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA context lease has no payload");
    }
    return context_lease;
  }

  // Acquire an owned context lease for the given device.
  static ContextLease acquireOwnedContext(DeviceHandle device) {
    auto device_lease = acquireDevice(device);
    auto *device_resource = device_lease.operator->();
    auto context_lease = device_resource->context_manager.acquireOwned();
    if (!context_lease.operator->()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA context lease has no payload");
    }
    return context_lease;
  }

  static KernelBaseLease acquireKernelBase(const KernelKeys &keys) {
    return manager().kernelBaseManager().acquire(keys);
  }

  static KernelMetadataLease acquireKernelMetadata(const KernelKeys &keys) {
    return manager().kernelMetadataManager().acquire(keys);
  }

private:
  // Singleton access to the execution manager (hidden from external callers).
  static ExecutionManager &manager() {
    static ExecutionManager instance{};
    return instance;
  }
};

} // namespace orteaf::internal::execution::cuda::api

#endif // ORTEAF_ENABLE_CUDA
