#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsComputePipelineStateManager::configure(const Config &config) {
  shutdown();
  if (config.device == nullptr || config.library == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS compute pipeline state manager requires a valid device and "
        "library");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS compute pipeline state manager requires valid ops");
  }
  device_ = config.device;
  library_ = config.library;
  ops_ = config.ops;
  key_to_index_.clear();

  const PipelinePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.configure(config.pool, payload_request, payload_context);
  core_.setConfigured(true);
}

void MpsComputePipelineStateManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  core_.checkCanShutdownOrThrow();
  const PipelinePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdownPayloadPool(payload_request, payload_context);
  core_.shutdownControlBlockPool();
  key_to_index_.clear();
  device_ = nullptr;
  library_ = nullptr;
  ops_ = nullptr;
}

MpsComputePipelineStateManager::PipelineLease
MpsComputePipelineStateManager::acquire(const FunctionKey &key) {
  core_.ensureConfigured();
  validateKey(key);

  // Check cache first
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    auto handle =
        FunctionHandle{static_cast<FunctionHandle::index_type>(it->second)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS compute pipeline state cache is invalid");
    }
    return core_.acquireWeakLease(handle);
  }

  // Reserve an uncreated slot and create the pipeline state
  PipelinePayloadPoolTraits::Request request{key};
  const auto context = makePayloadContext();
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS compute pipeline state manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS compute pipeline state");
  }

  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  return core_.acquireWeakLease(handle);
}

void MpsComputePipelineStateManager::validateKey(const FunctionKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Function identifier cannot be empty");
  }
}

PipelinePayloadPoolTraits::Context
MpsComputePipelineStateManager::makePayloadContext() const noexcept {
  return PipelinePayloadPoolTraits::Context{device_, library_, ops_};
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
