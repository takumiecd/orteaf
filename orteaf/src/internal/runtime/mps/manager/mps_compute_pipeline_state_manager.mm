#include <orteaf/internal/runtime/mps/manager/mps_compute_pipeline_state_manager.h>

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsComputePipelineStateManager::initialize(DeviceType device,
                                                LibraryType library,
                                                SlowOps *ops,
                                                std::size_t capacity) {
  shutdown();
  if (device == nullptr || library == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS compute pipeline state manager requires a valid device and "
        "library");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS compute pipeline state manager requires valid ops");
  }
  if (capacity > static_cast<std::size_t>(FunctionHandle::invalid_index())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS compute pipeline state manager capacity exceeds maximum handle "
        "range");
  }
  device_ = device;
  library_ = library;
  ops_ = ops;
  clearCacheStates();
  key_to_index_.clear();
  if (capacity > 0) {
    states_.reserve(capacity);
  }
  initialized_ = true;
}

void MpsComputePipelineStateManager::shutdown() {
  if (!initialized_) {
    return;
  }
  for (std::size_t i = 0; i < states_.size(); ++i) {
    State &state = states_[i];
    if (state.alive) {
      destroyResource(state.resource);
      state.alive = false;
    }
  }
  clearCacheStates();
  key_to_index_.clear();
  device_ = nullptr;
  library_ = nullptr;
  ops_ = nullptr;
  initialized_ = false;
}

MpsComputePipelineStateManager::PipelineLease
MpsComputePipelineStateManager::acquire(const FunctionKey &key) {
  ensureInitialized();
  validateKey(key);

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    incrementUseCount(it->second);
    return PipelineLease{this, createHandle<FunctionHandle>(it->second),
                         states_[it->second].resource.pipeline_state};
  }

  // Create new entry
  const std::size_t index = allocateSlot();
  State &state = states_[index];

  state.resource.function = ops_->createFunction(library_, key.identifier);
  if (state.resource.function == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS function for compute pipeline");
  }

  state.resource.pipeline_state =
      ops_->createComputePipelineState(device_, state.resource.function);
  if (state.resource.pipeline_state == nullptr) {
    ops_->destroyFunction(state.resource.function);
    state.resource.function = nullptr;
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS compute pipeline state");
  }

  markSlotAlive(index);
  key_to_index_.emplace(key, index);

  return PipelineLease{this, createHandle<FunctionHandle>(index),
                       state.resource.pipeline_state};
}

void MpsComputePipelineStateManager::release(PipelineLease &lease) noexcept {
  if (!lease) {
    return;
  }
  decrementUseCount(static_cast<std::size_t>(lease.handle().index));
  lease.invalidate();
}

void MpsComputePipelineStateManager::validateKey(const FunctionKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Function identifier cannot be empty");
  }
}

void MpsComputePipelineStateManager::destroyResource(
    MpsPipelineResource &resource) {
  if (resource.pipeline_state != nullptr) {
    ops_->destroyComputePipelineState(resource.pipeline_state);
    resource.pipeline_state = nullptr;
  }
  if (resource.function != nullptr) {
    ops_->destroyFunction(resource.function);
    resource.function = nullptr;
  }
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
