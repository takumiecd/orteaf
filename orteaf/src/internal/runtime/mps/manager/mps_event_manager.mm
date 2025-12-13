#include "orteaf/internal/runtime/mps/manager/mps_event_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsEventManager::initialize(DeviceType device, SlowOps *ops,
                                 std::size_t capacity) {
  shutdown();
  if (device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS event manager requires a valid device");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS event manager requires valid ops");
  }
  if (capacity > static_cast<std::size_t>(EventHandle::invalid_index())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS event manager capacity exceeds maximum handle range");
  }
  device_ = device;
  ops_ = ops;
  clearPoolStates();
  if (capacity > 0) {
    Base::growPool(capacity);
  }
  initialized_ = true;
}

void MpsEventManager::shutdown() {
  if (!initialized_) {
    return;
  }
  for (std::size_t i = 0; i < states_.size(); ++i) {
    State &state = states_[i];
    if (state.alive && state.resource != nullptr) {
      ops_->destroyEvent(state.resource);
      state.resource = nullptr;
      state.alive = false;
      state.in_use = false;
      state.ref_count.store(0, std::memory_order_relaxed);
    }
  }
  clearPoolStates();
  device_ = nullptr;
  ops_ = nullptr;
  initialized_ = false;
}

MpsEventManager::EventLease MpsEventManager::acquire() {
  ensureInitialized();
  const std::size_t index = Base::allocateSlot();
  State &state = states_[index];

  if (!state.alive) {
    state.resource = ops_->createEvent(device_);
    if (state.resource == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Failed to create MPS event");
    }
    state.alive = true;
    state.generation = 0;
  }

  markSlotInUse(index);
  return EventLease{this, createHandle<EventHandle>(index), state.resource};
}

MpsEventManager::EventLease MpsEventManager::acquire(EventHandle handle) {
  State &state = validateAndGetState(handle);
  incrementRefCount(static_cast<std::size_t>(handle.index));
  return EventLease{this, handle, state.resource};
}

void MpsEventManager::release(EventLease &lease) noexcept {
  release(lease.handle());
  lease.invalidate();
}

void MpsEventManager::release(EventHandle handle) noexcept {
  State *state = getStateForRelease(handle);
  if (state == nullptr) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(handle.index);
  const std::size_t new_count = decrementRefCount(index);
  if (new_count == 0) {
    releaseSlot(index);
  }
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
