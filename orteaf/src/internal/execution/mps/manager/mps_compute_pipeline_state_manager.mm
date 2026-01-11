#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsComputePipelineStateManager::configure(const InternalConfig &config) {
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
  const auto &cfg = config.public_config;
  key_to_index_.clear();

  const PipelinePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<PipelinePayloadPoolTraits::Request,
                PipelinePayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(
          cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext(payload_context)
      .configure(core_);
}

void MpsComputePipelineStateManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  lifetime_.clear();
  const PipelinePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
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
    auto cached = lifetime_.get(handle);
    if (cached) {
      return cached;
    }
    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
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
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
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
