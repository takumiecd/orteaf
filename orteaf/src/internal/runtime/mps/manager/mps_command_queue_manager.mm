#include "orteaf/internal/runtime/mps/manager/mps_command_queue_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsCommandQueueManager::initialize(DeviceType device, SlowOps *ops,
                                        std::size_t capacity) {
  shutdown();
  if (device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires a valid device");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires valid ops");
  }
  if (capacity >
      static_cast<std::size_t>(CommandQueueHandle::invalid_index())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager capacity exceeds maximum handle range");
  }
  device_ = device;
  ops_ = ops;

  const CommandQueuePayloadPoolTraits::Request request{};
  const CommandQueuePayloadPoolTraits::Context context{device_, ops_};
  core_.payloadPool().initializeAndCreate(
      CommandQueuePayloadPoolTraits::Config{capacity}, request, context);
  core_.initializeControlBlockPool(capacity);
  core_.setInitialized(true);
}

void MpsCommandQueueManager::shutdown() {
  if (!core_.isInitialized()) {
    return;
  }
  // Check canShutdown on all created control blocks
  core_.checkCanShutdownOrThrow();

  const CommandQueuePayloadPoolTraits::Request payload_request{};
  const CommandQueuePayloadPoolTraits::Context payload_context{device_, ops_};
  core_.payloadPool().shutdown(payload_request, payload_context);
  core_.shutdownControlBlockPool();

  device_ = nullptr;
  ops_ = nullptr;
  core_.setInitialized(false);
}

MpsCommandQueueManager::CommandQueueLease MpsCommandQueueManager::acquire() {
  core_.ensureInitialized();
  const CommandQueuePayloadPoolTraits::Request request{};
  const CommandQueuePayloadPoolTraits::Context context{device_, ops_};

  auto payload_ref = core_.payloadPool().tryAcquire(request, context);
  if (!payload_ref.valid()) {
    auto reserved = core_.payloadPool().tryReserve(request, context);
    if (!reserved.valid()) {
      const auto desired =
          core_.payloadPool().capacity() + core_.growthChunkSize();
      core_.payloadPool().grow(CommandQueuePayloadPoolTraits::Config{desired});
      reserved = core_.payloadPool().tryReserve(request, context);
    }
    if (!reserved.valid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
          "MPS command queue manager has no available slots");
    }
    CommandQueuePayloadPoolTraits::Request create_request{reserved.handle};
    if (!core_.payloadPool().emplace(reserved.handle, create_request,
                                     context)) {
      core_.payloadPool().release(reserved.handle);
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Failed to create MPS command queue");
    }
    payload_ref = reserved;
  }

  auto cb_ref = core_.acquireControlBlock();
  auto cb_handle = cb_ref.handle;
  auto *cb = cb_ref.payload_ptr;
  if (cb == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS command queue control block is unavailable");
  }

  auto *payload_ptr = core_.payloadPool().get(payload_ref.handle);
  if (payload_ptr == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS command queue payload is unavailable");
  }
  if (!cb->tryBindPayload(payload_ref.handle, payload_ptr,
                          &core_.payloadPool())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS command queue control block binding failed");
  }
  return CommandQueueLease{cb, &core_.controlBlockPool(), cb_handle};
}

MpsCommandQueueManager::CommandQueueLease
MpsCommandQueueManager::acquire(CommandQueueHandle handle) {
  core_.ensureInitialized();

  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid command queue handle");
  }

  auto *payload_ptr = core_.payloadPool().get(handle);
  if (payload_ptr == nullptr || *payload_ptr == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Command queue handle does not exist");
  }

  auto cb_ref = core_.acquireControlBlock();
  auto cb_handle = cb_ref.handle;
  auto *cb = cb_ref.payload_ptr;
  if (cb == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS command queue control block is unavailable");
  }

  if (!cb->tryBindPayload(handle, payload_ptr, &core_.payloadPool())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS command queue control block binding failed");
  }
  return CommandQueueLease{cb, &core_.controlBlockPool(), cb_handle};
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
