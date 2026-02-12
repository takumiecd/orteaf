#include "orteaf/internal/execution/cuda/manager/cuda_kernel_metadata_manager.h"

#if ORTEAF_ENABLE_CUDA

namespace orteaf::internal::execution::cuda::manager {

bool KernelMetadataPayloadPoolTraits::create(Payload &payload,
                                             const Request &request,
                                             const Context &) {
  const bool key_ok = payload.initialize(request.keys);
  payload.setExecute(request.execute);
  return key_ok;
}

void KernelMetadataPayloadPoolTraits::destroy(Payload &payload,
                                              const Request &, const Context &) {
  payload.reset();
  payload.setExecute(nullptr);
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
  request.execute = nullptr;
  const KernelMetadataPayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);
  return core_.acquireStrongLease(handle);
}

CudaKernelMetadataManager::CudaKernelMetadataLease
CudaKernelMetadataManager::acquire(
    const ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata
        &metadata) {
  core_.ensureConfigured();

  KernelMetadataPayloadPoolTraits::Request request{};
  request.keys = metadata.keys();
  request.execute = metadata.execute();
  const KernelMetadataPayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);
  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
