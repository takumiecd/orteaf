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

  // Initialize and pre-create resources
  Base::setupPool(capacity, [&](CommandQueueType &payload) {
    auto queue = ops_->createCommandQueue(device_);
    if (queue) {
      payload = queue;
    }
    return true;
  });
}

void MpsCommandQueueManager::shutdown() {
  if (!Base::isInitialized()) {
    return;
  }
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

// =============================================================================
// Acquire / Release
// =============================================================================

MpsCommandQueueManager::CommandQueueLease MpsCommandQueueManager::acquire() {
  ensureInitialized();

  auto createFn = [this](CommandQueueType &queue) {
    if (!queue) {
      if (!ops_) {
        return false;
      }
      queue = ops_->createCommandQueue(device_);
      if (!queue) {
        return false;
      }
    }
    return true;
  };

  auto handle = Base::acquireFresh(createFn);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS command queue");
  }

  return CommandQueueLease{handle, this};
}

void MpsCommandQueueManager::release(CommandQueueLease &lease) noexcept {
  if (lease.isValid()) {
    Base::releaseForReuse(lease.handle());
    lease.invalidate();
  }
}

// =============================================================================
// Locking API
// =============================================================================

MpsCommandQueueManager::ScopedLock
MpsCommandQueueManager::lock(const CommandQueueLease &lease) {
  if (!lease.isValid()) {
    return ScopedLock{}; // Invalid lock
  }
  auto &cb = Base::getControlBlockChecked(lease.handle());
  auto mutex_lock = cb.lock(); // Blocking lock
  return ScopedLock{std::move(mutex_lock), cb.payload()};
}

MpsCommandQueueManager::ScopedLock
MpsCommandQueueManager::tryLock(const CommandQueueLease &lease) {
  if (!lease.isValid()) {
    return ScopedLock{}; // Invalid lock
  }
  auto &cb = Base::getControlBlockChecked(lease.handle());
  auto mutex_lock = cb.tryLock(); // Non-blocking
  if (mutex_lock.owns_lock()) {
    return ScopedLock{std::move(mutex_lock), cb.payload()};
  }
  return ScopedLock{}; // Failed to acquire
}

// =============================================================================
// Internal
// =============================================================================

void MpsCommandQueueManager::destroyResource(CommandQueueType &resource) {
  if (resource) {
    if (ops_) {
      ops_->destroyCommandQueue(resource);
    }
    resource = nullptr;
  }
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
