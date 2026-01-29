#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/kernel/mps/mps_kernel_base_manager.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::kernel::mps {

// =============================================================================
// PayloadPoolTraits implementation
// =============================================================================

bool KernelBasePayloadPoolTraits::create(Payload &payload,
                                          const Request &request,
                                          const Context &context) {
  if (context.library_manager == nullptr || context.ops == nullptr) {
    return false;
  }

  // Acquire pipeline leases for each library/function key pair
  payload.pipelines.reserve(request.keys.size());
  for (const auto &key : request.keys) {
    // Get library lease
    auto library_lease = context.library_manager->acquire(key.first);
    if (!library_lease) {
      // Failed to acquire library, cleanup and return false
      payload.pipelines.clear();
      return false;
    }

    // Get library resource from the lease using operator->()
    auto *library_resource = library_lease.operator->();
    if (library_resource == nullptr) {
      payload.pipelines.clear();
      return false;
    }

    // Acquire pipeline lease from pipeline manager
    auto pipeline_lease = library_resource->pipeline_manager.acquire(key.second);
    if (!pipeline_lease) {
      payload.pipelines.clear();
      return false;
    }

    // Store the pipeline lease (library lease will be held by pipeline)
    payload.pipelines.pushBack(std::move(pipeline_lease));
  }

  return true;
}

void KernelBasePayloadPoolTraits::destroy(Payload &payload,
                                           const Request &,
                                           const Context &) {
  // Pipeline leases are automatically released on destruction
  payload.pipelines.clear();
}

// =============================================================================
// MpsKernelBaseManager implementation
// =============================================================================

void MpsKernelBaseManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.library_manager == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS kernel base manager requires a valid library manager");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS kernel base manager requires valid ops");
  }
  
  library_manager_ = config.library_manager;
  ops_ = config.ops;
  
  const auto &cfg = config.public_config;
  const KernelBasePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  
  Core::Builder<KernelBasePayloadPoolTraits::Request,
                KernelBasePayloadPoolTraits::Context>{}
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

void MpsKernelBaseManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  
  const KernelBasePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
  
  library_manager_ = nullptr;
  ops_ = nullptr;
}

MpsKernelBaseManager::KernelBaseLease
MpsKernelBaseManager::acquire(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  core_.ensureConfigured();
  
  KernelBasePayloadPoolTraits::Request request{};
  request.keys = keys;
  const auto context = makePayloadContext();
  
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS kernel base manager has no available slots");
  }
  
  return core_.acquireStrongLease(handle);
}

KernelBasePayloadPoolTraits::Context
MpsKernelBaseManager::makePayloadContext() const noexcept {
  KernelBasePayloadPoolTraits::Context context{};
  context.library_manager = library_manager_;
  context.ops = ops_;
  return context;
}

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
