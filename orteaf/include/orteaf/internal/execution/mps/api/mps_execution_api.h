#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <memory>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/mps/manager/mps_execution_manager.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution/mps/mps_handles.h"

namespace orteaf::internal::execution::mps::api {

class MpsExecutionApi {
public:
  using ExecutionManager =
      ::orteaf::internal::execution::mps::manager::MpsExecutionManager;
  using DeviceHandle = ::orteaf::internal::execution::mps::MpsDeviceHandle;
  using DeviceLease =
      ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease;
  using HeapDescriptorKey =
      ::orteaf::internal::execution::mps::manager::HeapDescriptorKey;
  using HeapLease =
      ::orteaf::internal::execution::mps::manager::MpsHeapManager::HeapLease;
  using HeapHandle = ::orteaf::internal::execution::mps::MpsHeapHandle;
  using LibraryKey = ::orteaf::internal::execution::mps::manager::LibraryKey;
  using FunctionKey = ::orteaf::internal::execution::mps::manager::FunctionKey;
  using KernelKey =
      ::orteaf::internal::execution::mps::manager::MpsKernelBaseManager::Key;
  using KernelKeys = ::orteaf::internal::base::HeapVector<KernelKey>;
  using KernelBaseLease = ::orteaf::internal::execution::mps::manager::
      MpsKernelBaseManager::KernelBaseLease;
  using KernelMetadataLease = ::orteaf::internal::execution::mps::manager::
      MpsKernelMetadataManager::KernelMetadataLease;
  using KernelExecuteFunc = ::orteaf::internal::execution::mps::manager::
      MpsKernelMetadataManager::ExecuteFunc;
  using PipelineLease = ::orteaf::internal::execution::mps::manager::
      MpsComputePipelineStateManager::PipelineLease;
  using StrongFenceLease = ::orteaf::internal::execution::mps::manager::
      MpsFenceManager::StrongFenceLease;
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;

  MpsExecutionApi() = delete;

  // Configure execution manager with default configuration.
  static void configure() { manager().configure(); }

  // Configure execution manager with the provided configuration.
  static void configure(const ExecutionManager::Config &config) {
    manager().configure(config);
  }

  static void shutdown() { manager().shutdown(); }

  static DeviceLease acquireDevice(DeviceHandle device) {
    auto device_lease = manager().deviceManager().acquire(device);
    if (!device_lease.operator->()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS device lease has no payload");
    }
    return device_lease;
  }

  static HeapLease acquireHeap(DeviceHandle device,
                               const HeapDescriptorKey &key) {
    auto device_lease = acquireDevice(device);
    return device_lease->heapManager().acquire(key);
  }

  static HeapLease acquireHeap(const HeapDescriptorKey &key) {
    return acquireHeap(DeviceHandle{0}, key);
  }

  static HeapLease acquireHeap(HeapHandle handle) {
    auto device_lease = acquireDevice(DeviceHandle{0});
    return device_lease->heapManager().acquire(handle);
  }

  // Acquire a single pipeline for the given device/library/function key trio.
  static PipelineLease acquirePipeline(DeviceHandle device,
                                       const LibraryKey &library_key,
                                       const FunctionKey &function_key) {
    auto device_lease = acquireDevice(device);
    auto *resource = device_lease.operator->();
    auto library_lease = resource->libraryManager().acquire(library_key);
    auto *library_resource = library_lease.operator->();
    if (library_resource == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS library lease has no payload");
    }
    return library_resource->pipelineManager().acquire(function_key);
  }

  static StrongFenceLease acquireFence(DeviceHandle device) {
    auto device_lease = acquireDevice(device);
    auto *resource = device_lease.operator->();
    return resource->fencePool().acquire();
  }

  static KernelBaseLease acquireKernelBase(const KernelKeys &keys) {
    return manager().kernelBaseManager().acquire(keys);
  }

  static KernelMetadataLease acquireKernelMetadata(const KernelKeys &keys,
                                                   KernelExecuteFunc execute) {
    return manager().kernelMetadataManager().acquire(keys, execute);
  }

private:
  // Singleton access to the execution manager (hidden from external callers).
  static ExecutionManager &manager() {
    static ExecutionManager instance{};
    return instance;
  }
};

} // namespace orteaf::internal::execution::mps::api

#endif // ORTEAF_ENABLE_MPS
