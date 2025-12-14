#include <orteaf/internal/runtime/mps/manager/mps_compute_pipeline_state_manager.h>

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsComputePipelineStateManager::initialize(DeviceType device,
                                                LibraryType library,
                                                SlowOps *ops,
                                                std::size_t capacity) {
  if (isInitialized()) {
    shutdown();
  }
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
  key_to_index_.clear();

  // Reserve capacity but don't initialize slots (lazy creation)
  reserveSlots(capacity);
}

void MpsComputePipelineStateManager::shutdown() {
  if (!isInitialized()) {
    return;
  }

  // Destroy all created resources
  shutdownSlots([this](auto &cb, auto handle) {
    if (cb.payload.isInitialized()) {
      destroyResource(cb.payload.get());
    }
  });

  key_to_index_.clear();
  device_ = nullptr;
  library_ = nullptr;
  ops_ = nullptr;
}

MpsComputePipelineStateManager::PipelineLease
MpsComputePipelineStateManager::acquire(const FunctionKey &key) {
  ensureInitialized();
  validateKey(key);

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    auto handle = static_cast<FunctionHandle>(it->second);
    auto &cb = getControlBlock(handle);
    return PipelineLease{this, handle, cb.payload.get().pipeline_state};
  }

  // Need new slot - pop from freelist
  Handle handle;
  if (!tryPopFromFreelist(handle)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "No slots available for new pipeline state");
  }

  auto &cb = getControlBlock(handle);
  auto &resource = cb.payload.get();

  // Create function
  resource.function = ops_->createFunction(library_, key.identifier);
  if (resource.function == nullptr) {
    pushToFreelist(handle);
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS function for compute pipeline");
  }

  // Create pipeline state
  resource.pipeline_state =
      ops_->createComputePipelineState(device_, resource.function);
  if (resource.pipeline_state == nullptr) {
    ops_->destroyFunction(resource.function);
    resource.function = nullptr;
    pushToFreelist(handle);
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS compute pipeline state");
  }

  cb.payload.markInitialized();
  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));

  return PipelineLease{this, handle, resource.pipeline_state};
}

void MpsComputePipelineStateManager::release(PipelineLease &lease) noexcept {
  if (!lease) {
    return;
  }
  // RawLease: no ref counting, just invalidate
  // Resource stays in cache until shutdown
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
