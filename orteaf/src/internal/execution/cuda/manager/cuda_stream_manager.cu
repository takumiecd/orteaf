#include "orteaf/internal/execution/cuda/manager/cuda_stream_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

void CudaStreamManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA stream manager requires a valid context");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA stream manager requires valid ops");
  }
  context_ = config.context;
  ops_ = config.ops;
  const auto &cfg = config.public_config;

  const StreamPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<StreamPayloadPoolTraits::Request,
                StreamPayloadPoolTraits::Context>{}
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

void CudaStreamManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  const StreamPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
  context_ = nullptr;
  ops_ = nullptr;
}

CudaStreamManager::StreamLease CudaStreamManager::acquire() {
  core_.ensureConfigured();
  const StreamPayloadPoolTraits::Request request{};
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA stream manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

CudaStreamManager::StreamLease
CudaStreamManager::acquire(StreamHandle handle) {
  core_.ensureConfigured();

  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid stream handle");
  }

  if (!core_.isAlive(handle)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "Stream handle does not correspond to an alive payload");
  }

  return core_.acquireStrongLease(handle);
}

StreamPayloadPoolTraits::Context
CudaStreamManager::makePayloadContext() const noexcept {
  return StreamPayloadPoolTraits::Context{context_, ops_};
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
