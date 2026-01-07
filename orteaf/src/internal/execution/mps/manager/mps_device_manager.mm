#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::execution::mps::manager {

void MpsDeviceManager::configure(const Config &config) {
  shutdown();

  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS device manager requires valid ops");
  }
  ops_ = config.ops;
  const int device_count = ops_->getDeviceCount();
  const std::size_t device_count_size =
      device_count <= 0 ? 0u : static_cast<std::size_t>(device_count);
  const std::size_t payload_capacity = config.pool.payload_capacity;
  if (payload_capacity != device_count_size) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS device manager payload size does not match device count");
  }
  const DevicePayloadPoolTraits::Request payload_request{};
  const auto payload_context =
      DevicePayloadPoolTraits::Context{ops_,
                                       config.command_queue_config,
                                       config.event_config,
                                       config.fence_config,
                                       config.heap_config,
                                       config.library_config,
                                       config.graph_config};
  if (device_count <= 0) {
    core_.configure(config.pool, payload_request, payload_context);
    return;
  }

  core_.configure(config.pool, payload_request, payload_context);
  core_.createAllPayloads(payload_request, payload_context);
}

void MpsDeviceManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  // Check canShutdown on all created control blocks
  lifetime_.clear();

  const DevicePayloadPoolTraits::Request payload_request{};
  const auto payload_context =
      DevicePayloadPoolTraits::Context{ops_,
                                       MpsCommandQueueManager::Config{},
                                       MpsEventManager::Config{},
                                       MpsFenceManager::Config{},
                                       MpsHeapManager::Config{},
                                       MpsLibraryManager::Config{},
                                       MpsGraphManager::Config{}};
  core_.shutdown(payload_request, payload_context);
  ops_ = nullptr;
}

MpsDeviceManager::DeviceLease MpsDeviceManager::acquire(DeviceHandle handle) {
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
