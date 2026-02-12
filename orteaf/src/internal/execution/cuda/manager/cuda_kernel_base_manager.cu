#include "orteaf/internal/execution/cuda/manager/cuda_kernel_base_manager.h"

#if ORTEAF_ENABLE_CUDA

namespace orteaf::internal::execution::cuda::manager {

bool KernelBasePayloadPoolTraits::create(Payload &payload,
                                         const Request &request,
                                         const Context &) {
  const bool key_ok = payload.setKeys(request.keys);
  payload.setExecute(request.execute);
  return key_ok;
}

void KernelBasePayloadPoolTraits::destroy(Payload &payload, const Request &,
                                          const Context &) {
  payload.reset();
}

void CudaKernelBaseManager::configure(const InternalConfig &config) {
  shutdown();

  const auto &cfg = config.public_config;
  const KernelBasePayloadPoolTraits::Request payload_request{};
  const KernelBasePayloadPoolTraits::Context payload_context{};

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

void CudaKernelBaseManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  const KernelBasePayloadPoolTraits::Request payload_request{};
  const KernelBasePayloadPoolTraits::Context payload_context{};
  core_.shutdown(payload_request, payload_context);
}

CudaKernelBaseManager::KernelBaseLease CudaKernelBaseManager::acquire(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  core_.ensureConfigured();

  KernelBasePayloadPoolTraits::Request request{};
  request.keys = keys;
  request.execute = nullptr;
  const KernelBasePayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);
  return core_.acquireStrongLease(handle);
}

CudaKernelBaseManager::KernelBaseLease CudaKernelBaseManager::acquire(
    const ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata
        &metadata) {
  core_.ensureConfigured();

  KernelBasePayloadPoolTraits::Request request{};
  request.keys = metadata.keys();
  request.execute = metadata.execute();
  const KernelBasePayloadPoolTraits::Context context{};

  auto handle = core_.reserveUncreatedPayloadOrGrowAndEmplaceOrThrow(request,
                                                                     context);
  return core_.acquireStrongLease(handle);
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
