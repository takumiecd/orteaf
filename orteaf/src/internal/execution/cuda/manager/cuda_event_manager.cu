#include "orteaf/internal/execution/cuda/manager/cuda_event_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

void CudaEventManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA event manager requires a valid context");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA event manager requires valid ops");
  }
  context_ = config.context;
  ops_ = config.ops;
  const auto &cfg = config.public_config;

  const EventPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<EventPayloadPoolTraits::Request,
                EventPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext(payload_context)
      .configure(core_);
}

void CudaEventManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  const EventPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
  context_ = nullptr;
  ops_ = nullptr;
}

CudaEventManager::EventLease CudaEventManager::acquire() {
  core_.ensureConfigured();
  const EventPayloadPoolTraits::Request request{};
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA event manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

EventPayloadPoolTraits::Context
CudaEventManager::makePayloadContext() const noexcept {
  return EventPayloadPoolTraits::Context{context_, ops_};
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
