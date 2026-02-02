#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/manager/mps_kernel_metadata_manager.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

// =============================================================================
// PayloadPoolTraits implementation
// =============================================================================

bool KernelMetadataPayloadPoolTraits::create(Payload &payload,
                                             const Request &request,
                                             const Context &) {
  return payload.initialize(request.keys);
}

void KernelMetadataPayloadPoolTraits::destroy(Payload &payload,
                                              const Request &,
                                              const Context &) {
  payload.reset();
}

// =============================================================================
// MpsKernelMetadataManager implementation
// =============================================================================

void MpsKernelMetadataManager::configure(const InternalConfig &config) {
  shutdown();

  const auto &cfg = config.public_config;
  const KernelMetadataPayloadPoolTraits::Request payload_request{};
  const KernelMetadataPayloadPoolTraits::Context payload_context{};

  Core::Builder<KernelMetadataPayloadPoolTraits::Request,
                KernelMetadataPayloadPoolTraits::Context>{}
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

void MpsKernelMetadataManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  const KernelMetadataPayloadPoolTraits::Request payload_request{};
  const KernelMetadataPayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

MpsKernelMetadataManager::MpsKernelMetadataLease
MpsKernelMetadataManager::acquire(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  core_.ensureConfigured();

  KernelMetadataPayloadPoolTraits::Request request{};
  request.keys = keys;

  const KernelMetadataPayloadPoolTraits::Context context{};
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS kernel metadata manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS kernel metadata manager failed to create payload");
  }

  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
