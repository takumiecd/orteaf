#include "orteaf/internal/execution/mps/manager/mps_command_queue_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsCommandQueueManager::configure(const Config &config) {
  shutdown();
  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  // payload block size managed by core_
  // payload growth chunk size configured via core_

  const CommandQueuePayloadPoolTraits::Request payload_request{};
  const CommandQueuePayloadPoolTraits::Context payload_context{device_, ops_};
  core_.configure(config.pool, payload_request, payload_context);
  if (!core_.createAllPayloads(payload_request, payload_context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS command queues");
  }
}

void MpsCommandQueueManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  // Check canShutdown on all created control blocks
  core_.checkCanShutdownOrThrow();

  const CommandQueuePayloadPoolTraits::Request payload_request{};
  const CommandQueuePayloadPoolTraits::Context payload_context{device_, ops_};
  core_.shutdownPayloadPool(payload_request, payload_context);
  core_.shutdownControlBlockPool();

  device_ = nullptr;
  ops_ = nullptr;
}

MpsCommandQueueManager::CommandQueueLease MpsCommandQueueManager::acquire() {
  core_.ensureConfigured();
  const CommandQueuePayloadPoolTraits::Request request{};
  const CommandQueuePayloadPoolTraits::Context context{device_, ops_};

  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS command queue manager has no available slots");
  }
  return core_.acquireWeakLease(handle);
}

MpsCommandQueueManager::CommandQueueLease
MpsCommandQueueManager::acquire(CommandQueueHandle handle) {
  core_.ensureConfigured();

  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid command queue handle");
  }

  auto lease = core_.acquireWeakLease(handle);
  const auto *payload_ptr = lease.payloadPtr();
  if (payload_ptr == nullptr || *payload_ptr == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Command queue handle does not exist");
  }
  return lease;
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
