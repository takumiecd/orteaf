#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"

namespace orteaf::internal::execution::mps::manager {

using DeviceLease = ::orteaf::internal::execution::mps::manager::
    MpsDeviceManager::DeviceLease;

// =============================================================================
// PayloadPoolTraits implementation
// =============================================================================

bool KernelBasePayloadPoolTraits::create(Payload &payload,
                                        const Request &request,
                                        const Context &) {
  return payload.setKeys(request.keys);
}

void KernelBasePayloadPoolTraits::destroy(Payload &payload,
                                           const Request &,
                                           const Context &) {
  payload.reset();
}

// =============================================================================
// MpsKernelBaseManager implementation
// =============================================================================

void MpsKernelBaseManager::configure(const InternalConfig &config) {
  shutdown();
  
  const auto &cfg = config.public_config;
  const KernelBasePayloadPoolTraits::Request payload_request{};
  // Note: payload_context requires device_lease, set during acquire()
  
  Core::Builder<KernelBasePayloadPoolTraits::Request,
                KernelBasePayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext({}) // Context set per-acquire
      .configure(core_);
}

void MpsKernelBaseManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  
  const KernelBasePayloadPoolTraits::Request payload_request{};
  const KernelBasePayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

MpsKernelBaseManager::KernelBaseLease
MpsKernelBaseManager::acquire(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  core_.ensureConfigured();
  
  KernelBasePayloadPoolTraits::Request request{};
  request.keys = keys;
  const KernelBasePayloadPoolTraits::Context context{};
  
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS kernel base manager has no available slots");
  }
  
  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
