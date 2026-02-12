#include "orteaf/internal/execution/cpu/manager/cpu_kernel_base_manager.h"

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cpu::manager {

// =============================================================================
// Payload Pool Traits Implementation
// =============================================================================

bool KernelBasePayloadPoolTraits::create(Payload &payload,
                                         const Request &request,
                                         const Context & /*context*/) {
  // Set ExecuteFunc on the payload
  payload.setExecute(request.execute);
  return true;
}

void KernelBasePayloadPoolTraits::destroy(Payload &payload,
                                          const Request & /*request*/,
                                          const Context & /*context*/) {
  // Clear ExecuteFunc
  payload.setExecute(nullptr);
}

// =============================================================================
// CpuKernelBaseManager Implementation
// =============================================================================

void CpuKernelBaseManager::configure(const InternalConfig &config) {
  shutdown();

  const auto &cfg = config.public_config;
  const KernelBasePayloadPoolTraits::Request payload_request{};

  Core::Builder<KernelBasePayloadPoolTraits::Request,
                KernelBasePayloadPoolTraits::Context>{}
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

void CpuKernelBaseManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  const KernelBasePayloadPoolTraits::Request payload_request{};
  const KernelBasePayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

CpuKernelBaseManager::KernelBaseLease
CpuKernelBaseManager::acquire(ExecuteFunc execute) {
  core_.ensureConfigured();

  if (!execute) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        "ExecuteFunc must be valid");
  }

  KernelBasePayloadPoolTraits::Request request{};
  request.execute = execute;
  const KernelBasePayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);

  return core_.acquireStrongLease(handle);
}

CpuKernelBaseManager::KernelBaseLease
CpuKernelBaseManager::acquire(
    const ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata
        &metadata) {
  return acquire(metadata.execute());
}

} // namespace orteaf::internal::execution::cpu::manager
