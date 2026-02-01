#include "orteaf/internal/execution/cpu/manager/cpu_kernel_metadata_manager.h"

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cpu::manager {

// =============================================================================
// Payload Pool Traits Implementation
// =============================================================================

bool KernelMetadataPayloadPoolTraits::create(Payload &payload,
                                             const Request &request,
                                             const Context & /*context*/) {
  payload.setExecute(request.execute);
  return true;
}

void KernelMetadataPayloadPoolTraits::destroy(Payload &payload,
                                              const Request & /*request*/,
                                              const Context & /*context*/) {
  payload.setExecute(nullptr);
}

// =============================================================================
// CpuKernelMetadataManager Implementation
// =============================================================================

void CpuKernelMetadataManager::configure(const InternalConfig &config) {
  shutdown();

  const auto &cfg = config.public_config;
  const KernelMetadataPayloadPoolTraits::Request payload_request{};

  Core::Builder<KernelMetadataPayloadPoolTraits::Request,
                KernelMetadataPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext({})
      .configure(core_);
}

void CpuKernelMetadataManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  const KernelMetadataPayloadPoolTraits::Request payload_request{};
  const KernelMetadataPayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

CpuKernelMetadataManager::CpuKernelMetadataLease
CpuKernelMetadataManager::acquire(ExecuteFunc execute) {
  core_.ensureConfigured();

  if (!execute) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        "ExecuteFunc must be valid");
  }

  KernelMetadataPayloadPoolTraits::Request request{};
  request.execute = execute;
  const KernelMetadataPayloadPoolTraits::Context context{};

  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CPU kernel metadata manager has no available slots");
  }

  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::cpu::manager
