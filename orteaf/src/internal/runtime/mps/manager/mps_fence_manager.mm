#include "orteaf/internal/runtime/mps/manager/mps_fence_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsFenceManager::initialize(DeviceType device, SlowOps *ops,
                                 std::size_t capacity) {
  shutdown();
  if (device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS fence manager requires a valid device");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS fence manager requires valid ops");
  }
  if (capacity > static_cast<std::size_t>(FenceHandle::invalid_index())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS fence manager capacity exceeds maximum handle range");
  }
  device_ = device;
  ops_ = ops;
  states_.clear();
  Base::free_list_.clear();
  if (capacity > 0) {
    Base::growPool(capacity);
  }
  initialized_ = true;
}

void MpsFenceManager::shutdown() {
  if (!initialized_) {
    return;
  }
  for (std::size_t i = 0; i < states_.size(); ++i) {
    State &state = states_[i];
    if (state.alive && state.resource != nullptr) {
      ops_->destroyFence(state.resource);
      state.resource = nullptr;
      state.alive = false;
      state.in_use = false;
      state.ref_count.store(0, std::memory_order_relaxed);
    }
  }
  states_.clear();
  Base::free_list_.clear();
  device_ = nullptr;
  ops_ = nullptr;
  initialized_ = false;
}

MpsFenceManager::FenceLease MpsFenceManager::acquire() {
  ensureInitialized();
  const std::size_t index = Base::allocateSlot();
  State &state = states_[index];

  if (!state.alive) {
    state.resource = ops_->createFence(device_);
    if (state.resource == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Failed to create MPS fence");
    }
    state.alive = true;
    state.generation = 0;
  }

  markSlotInUse(index);

  const auto handle =
      FenceHandle{static_cast<FenceHandle::index_type>(index),
                  static_cast<FenceHandle::generation_type>(state.generation)};
  return FenceLease{this, handle, state.resource};
}

MpsFenceManager::FenceLease MpsFenceManager::acquire(FenceHandle handle) {
  ensureInitialized();
  const std::size_t index = static_cast<std::size_t>(handle.index);
  if (index >= states_.size()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS fence manager handle out of range");
  }
  State &state = states_[index];
  if (!state.alive || !state.in_use) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS fence manager handle is inactive");
  }
  if (!isGenerationValid(index, handle)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS fence manager handle is stale");
  }

  incrementRefCount(index);
  return FenceLease{this, handle, state.resource};
}

void MpsFenceManager::release(FenceLease &lease) noexcept {
  release(lease.handle());
  lease.invalidate();
}

void MpsFenceManager::release(FenceHandle handle) noexcept {
  if (!initialized_ || device_ == nullptr || ops_ == nullptr) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(handle.index);
  if (index >= states_.size()) {
    return;
  }
  State &state = states_[index];
  if (!state.alive || !state.in_use) {
    return;
  }
  if (!isGenerationValid(index, handle)) {
    return;
  }

  const std::size_t new_count = decrementRefCount(index);
  if (new_count == 0) {
    releaseSlot(index);
  }
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
