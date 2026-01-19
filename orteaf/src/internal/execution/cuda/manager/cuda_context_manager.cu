#include "orteaf/internal/execution/cuda/manager/cuda_context_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

void CudaContextManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA context manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  const auto &cfg = config.public_config;
  buffer_config_ = cfg.buffer_config;
  stream_config_ = cfg.stream_config;
  event_config_ = cfg.event_config;
  module_config_ = cfg.module_config;

  ContextPayloadPoolTraits::Request payload_request{};
  payload_request.kind = ContextKind::Primary;
  const auto payload_context = makePayloadContext();
  Core::Builder<ContextPayloadPoolTraits::Request,
                ContextPayloadPoolTraits::Context>{}
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

void CudaContextManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  const ContextPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
  device_ = DeviceType{};
  ops_ = nullptr;
  buffer_config_ = {};
  stream_config_ = {};
  event_config_ = {};
  module_config_ = {};
}

CudaContextManager::ContextLease CudaContextManager::acquirePrimary() {
  core_.ensureConfigured();
  ContextPayloadPoolTraits::Request request{};
  request.kind = ContextKind::Primary;
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA context manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

CudaContextManager::ContextLease CudaContextManager::acquireOwned() {
  core_.ensureConfigured();
  ContextPayloadPoolTraits::Request request{};
  request.kind = ContextKind::Owned;
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA context manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

ContextPayloadPoolTraits::Context
CudaContextManager::makePayloadContext() const noexcept {
  ContextPayloadPoolTraits::Context context{};
  context.device = device_;
  context.ops = ops_;
  context.buffer_config = buffer_config_;
  context.stream_config = stream_config_;
  context.event_config = event_config_;
  context.module_config = module_config_;
  return context;
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
