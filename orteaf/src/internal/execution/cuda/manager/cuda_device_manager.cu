#include "orteaf/internal/execution/cuda/manager/cuda_device_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

// =============================================================================
// DevicePayloadPoolTraits Implementation
// =============================================================================

bool DevicePayloadPoolTraits::create(Payload &payload, const Request &request,
                                     const Context &context) {
  if (context.ops == nullptr || !request.handle.isValid()) {
    return false;
  }

  const auto device = context.ops->getDevice(request.handle.index);
  payload.device = device;

  const auto capability = context.ops->getComputeCapability(device);
  const int cc_value = capability.major * 10 + capability.minor;
  auto vendor = context.ops->getDeviceVendor(device);
  if (vendor.empty()) {
    vendor = "nvidia";
  }
  payload.arch = ::orteaf::internal::architecture::detectCudaArchitecture(
      cc_value, vendor);

  CudaContextManager::InternalConfig context_config{};
  context_config.public_config = context.context_config;
  context_config.device = device;
  context_config.ops = context.ops;
  payload.context_manager.configure(context_config);

  return true;
}

void DevicePayloadPoolTraits::destroy(Payload &payload, const Request &,
                                      const Context &context) {
  payload.reset(context.ops);
}

// =============================================================================
// CudaDeviceManager Implementation
// =============================================================================

void CudaDeviceManager::configure(const InternalConfig &config) {
  shutdown();

  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA device manager requires valid ops");
  }

  ops_ = config.ops;
  const auto &cfg = config.public_config;
  const int device_count = ops_->getDeviceCount();
  const std::size_t device_count_size =
      device_count <= 0 ? 0u : static_cast<std::size_t>(device_count);
  const std::size_t payload_capacity = cfg.payload_capacity;

  if (payload_capacity != device_count_size) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA device manager payload size does not match device count");
  }

  const DevicePayloadPoolTraits::Request payload_request{};
  DevicePayloadPoolTraits::Context payload_context{};
  payload_context.ops = ops_;
  payload_context.context_config = cfg.context_config;

  Core::Builder<DevicePayloadPoolTraits::Request,
                DevicePayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext(payload_context)
      .configure(core_);

  if (device_count <= 0) {
    return;
  }

  if (!core_.createAllPayloads(payload_request, payload_context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create CUDA device payloads");
  }
}

void CudaDeviceManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  lifetime_.clear();

  const DevicePayloadPoolTraits::Request payload_request{};
  DevicePayloadPoolTraits::Context payload_context{};
  payload_context.ops = ops_;
  core_.shutdown(payload_request, payload_context);
  ops_ = nullptr;
}

CudaDeviceManager::DeviceLease CudaDeviceManager::acquire(DeviceHandle handle) {
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
