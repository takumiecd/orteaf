#include "orteaf/internal/execution/cuda/manager/cuda_kernel_metadata_manager.h"

#if ORTEAF_ENABLE_CUDA

namespace orteaf::internal::execution::cuda::manager {

bool KernelMetadataPayloadPoolTraits::create(Payload &payload,
                                             const Request &request,
                                             const Context &) {
  return payload.initialize(request.keys);
}

void KernelMetadataPayloadPoolTraits::destroy(Payload &payload,
                                              const Request &, const Context &) {
  payload.reset();
}

void CudaKernelMetadataManager::configure(const InternalConfig &config) {
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

void CudaKernelMetadataManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  const KernelMetadataPayloadPoolTraits::Request payload_request{};
  const KernelMetadataPayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

CudaKernelMetadataManager::CudaKernelMetadataLease
CudaKernelMetadataManager::acquire(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  core_.ensureConfigured();

  KernelMetadataPayloadPoolTraits::Request request{};
  request.keys = keys;
  const KernelMetadataPayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);
  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
