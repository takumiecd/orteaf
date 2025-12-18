#include <orteaf/internal/runtime/mps/manager/mps_command_queue_manager.h>

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsCommandQueueManager::initialize(DeviceType device, SlowOps *ops,
                                        std::size_t capacity) {
  shutdown();
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

  // Initialize and pre-create resources (lazy creation is also supported, but
  // this matches previous implementation of pre-creation if capacity > 0)
  Base::setupPool(capacity, [&](CommandQueueType &payload) {
    auto queue = ops_->createCommandQueue(device_);
    if (queue) {
      payload = queue;
    }
    // Always return true to add to freelist (in_use defaults to false)
    return true;
  });
}

void MpsCommandQueueManager::shutdown() {
  if (!Base::isInitialized()) {
    return;
  }
  // Cleanup all resources
  Base::teardownPool([this](CommandQueueType &payload) {
    if (payload != nullptr) {
      destroyResource(payload);
      payload = nullptr;
    }
  });
  device_ = nullptr;
  ops_ = nullptr;
}

void MpsCommandQueueManager::growCapacity(std::size_t additional) {
  Base::expandPool(additional, [this](CommandQueueType &payload) {
    payload = ops_->createCommandQueue(device_);
    return payload != nullptr;
  });
}

MpsCommandQueueManager::CommandQueueLease MpsCommandQueueManager::acquire() {
  auto handle = Base::acquireFresh([this](CommandQueueType &payload) {
    // If already pre-created during initialize/growCapacity, skip creation
    if (payload != nullptr) {
      return true;
    }
    if (!ops_)
      return false;
    auto queue = ops_->createCommandQueue(device_);
    if (!queue)
      return false;
    payload = queue;
    return true;
  });

  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS command queue");
  }

  return CommandQueueLease{this, handle,
                           Base::getControlBlockChecked(handle).payload()};
}

void MpsCommandQueueManager::release(CommandQueueLease &lease) noexcept {
  if (!lease) {
    return;
  }
  auto handle = lease.handle();
  lease.invalidate();
  if (!Base::isValidHandle(handle)) {
    return;
  }
  Base::releaseForReuse(handle);
}

// =============================================================================
// Weak Reference Support
// =============================================================================

MpsCommandQueueManager::CommandQueueWeakLease
MpsCommandQueueManager::acquireWeak(const CommandQueueLease &lease) {
  if (!lease) {
    return CommandQueueWeakLease{};
  }
  Base::getControlBlockChecked(lease.handle()).acquireWeak();
  return CommandQueueWeakLease{this, lease.handle()};
}

MpsCommandQueueManager::CommandQueueWeakLease
MpsCommandQueueManager::acquireWeak(CommandQueueHandle handle) {
  if (!Base::isValidHandle(handle)) {
    return CommandQueueWeakLease{};
  }
  Base::getControlBlockChecked(handle).acquireWeak();
  return CommandQueueWeakLease{this, handle};
}

void MpsCommandQueueManager::addWeakRef(CommandQueueHandle handle) noexcept {
  if (Base::isValidHandle(handle)) {
    Base::getControlBlockChecked(handle).acquireWeak();
  }
}

void MpsCommandQueueManager::dropWeakRef(
    CommandQueueWeakLease &lease) noexcept {
  if (!lease) {
    return;
  }
  auto handle = lease.handle();
  if (Base::isValidHandle(handle)) {
    Base::getControlBlockChecked(handle).releaseWeak();
  }
  lease.invalidate();
}

MpsCommandQueueManager::CommandQueueLease
MpsCommandQueueManager::tryPromote(CommandQueueHandle handle) {
  if (!Base::isValidHandle(handle)) {
    return CommandQueueLease{};
  }
  auto &cb = Base::getControlBlockChecked(handle);
  if (cb.tryPromote()) {
    return CommandQueueLease{this, handle, cb.payload()};
  }
  return CommandQueueLease{};
}

void MpsCommandQueueManager::destroyResource(CommandQueueType &resource) {
  if (resource != nullptr) {
    if (ops_) {
      ops_->destroyCommandQueue(resource);
    }
    resource = nullptr;
  }
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
