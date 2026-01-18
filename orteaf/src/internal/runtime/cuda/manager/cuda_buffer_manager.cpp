#include "orteaf/internal/runtime/cuda/manager/cuda_buffer_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::cuda::manager {

void CudaBufferManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA buffer manager requires a valid context");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA buffer manager requires valid ops");
  }

  context_ = config.context;
  ops_ = config.ops;
  alloc_ = config.public_config.alloc;
  free_ = config.public_config.free;
  if (alloc_ == nullptr) {
    alloc_ = &BufferPayloadPoolTraits::Resource::allocate;
  }
  if (free_ == nullptr) {
    free_ = &BufferPayloadPoolTraits::Resource::deallocate;
  }
  const auto &cfg = config.public_config;

  std::size_t payload_capacity = cfg.payload_capacity;
  if (payload_capacity == 0) {
    payload_capacity = 64;
  }
  std::size_t payload_block_size = cfg.payload_block_size;
  if (payload_block_size == 0) {
    payload_block_size = 16;
  }
  std::size_t control_block_capacity = cfg.control_block_capacity;
  if (control_block_capacity == 0) {
    control_block_capacity = 64;
  }
  std::size_t control_block_block_size = cfg.control_block_block_size;
  if (control_block_block_size == 0) {
    control_block_block_size = 16;
  }

  BufferPayloadPoolTraits::Request request{};
  BufferPayloadPoolTraits::Context context{};
  context.context = context_;
  context.ops = ops_;
  context.alloc = alloc_;
  context.free = free_;

  Core::Builder<BufferPayloadPoolTraits::Request,
                BufferPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(control_block_capacity)
      .withControlBlockBlockSize(control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(payload_capacity)
      .withPayloadBlockSize(payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(request)
      .withContext(context)
      .configure(core_);
}

void CudaBufferManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  BufferPayloadPoolTraits::Request request{};
  const auto context = makePayloadContext();
  core_.shutdown(request, context);
  context_ = nullptr;
  ops_ = nullptr;
  alloc_ = nullptr;
  free_ = nullptr;
}

CudaBufferManager::BufferLease CudaBufferManager::acquire(std::size_t size,
                                                          std::size_t alignment) {
  core_.ensureConfigured();
  if (size == 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA buffer manager requires size > 0");
  }
  BufferPayloadPoolTraits::Request request{};
  request.size = size;
  request.alignment = alignment;
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA buffer manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

BufferPayloadPoolTraits::Context
CudaBufferManager::makePayloadContext() const noexcept {
  return BufferPayloadPoolTraits::Context{context_, ops_, alloc_, free_};
}

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
