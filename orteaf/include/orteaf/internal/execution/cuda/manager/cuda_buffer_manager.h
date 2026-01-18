#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/lease/strong_lease.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/resource/cuda_buffer.h"
#include "orteaf/internal/execution/allocator/resource/cuda/cuda_resource.h"

namespace orteaf::internal::execution::cuda::manager {

struct ContextPayloadPoolTraits;

// =============================================================================
// Payload Pool Traits
// =============================================================================

template <typename ResourceT> struct BufferPayloadPoolTraitsT {
  using Payload = ::orteaf::internal::execution::cuda::resource::CudaBuffer;
  using Handle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  using BufferView =
      ::orteaf::internal::execution::cuda::resource::CudaBuffer::BufferView;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using Resource = ResourceT;

  struct Request {
    std::size_t size{0};
    std::size_t alignment{0};
    Handle handle{Handle::invalid()};
  };

  struct Context {
    ContextType context{nullptr};
    SlowOps *ops{nullptr};
    Resource *resource{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || context.context == nullptr ||
        context.resource == nullptr ||
        !request.handle.isValid()) {
      return false;
    }

    if (request.size == 0) {
      payload = Payload{};
      return true;
    }

    context.ops->setContext(context.context);
    auto view = context.resource->allocate(request.size, request.alignment);
    if (!view) {
      return false;
    }

    const auto view_handle =
        ::orteaf::internal::execution::cuda::CudaBufferViewHandle{
            static_cast<::orteaf::internal::execution::cuda::CudaBufferViewHandle::
                            underlying_type>(request.handle.index)};
    payload = Payload{view_handle, view};
    return true;
  }

  static void destroy(Payload &payload, const Request &request,
                      const Context &context) {
    if (!payload.valid()) {
      return;
    }
    if (context.ops != nullptr && context.context != nullptr &&
        context.resource != nullptr) {
      context.ops->setContext(context.context);
      context.resource->deallocate(payload.view, request.size,
                                   request.alignment);
    }
    payload = Payload{};
  }
};

// =============================================================================
// Payload Pool
// =============================================================================

template <typename ResourceT>
using BufferPayloadPoolT =
    ::orteaf::internal::base::pool::SlotPool<
        BufferPayloadPoolTraitsT<ResourceT>>;

struct BufferManagerCBTag {};

// =============================================================================
// ControlBlock
// =============================================================================

template <typename ResourceT>
using BufferControlBlockT = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaBufferHandle,
    ::orteaf::internal::execution::cuda::resource::CudaBuffer,
    BufferPayloadPoolT<ResourceT>>;

// =============================================================================
// Manager Traits
// =============================================================================

template <typename ResourceT> struct CudaBufferManagerTraits {
  using PayloadPool = BufferPayloadPoolT<ResourceT>;
  using ControlBlock = BufferControlBlockT<ResourceT>;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  static constexpr const char *Name = "CUDA buffer manager";
};

// =============================================================================
// CudaBufferManager
// =============================================================================

template <typename ResourceT> class CudaBufferManagerT {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using BufferHandle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  using BufferView =
      ::orteaf::internal::execution::cuda::resource::CudaBuffer::BufferView;
  using Resource = ResourceT;

  using Traits = CudaBufferManagerTraits<ResourceT>;
  using Core = ::orteaf::internal::base::PoolManager<Traits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using BufferLease = Core::StrongLeaseType;

  struct Config {
    typename Resource::Config resource_config{};
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaBufferManagerT() = default;
  CudaBufferManagerT(const CudaBufferManagerT &) = delete;
  CudaBufferManagerT &operator=(const CudaBufferManagerT &) = delete;
  CudaBufferManagerT(CudaBufferManagerT &&) = default;
  CudaBufferManagerT &operator=(CudaBufferManagerT &&) = default;
  ~CudaBufferManagerT() = default;

private:
  struct InternalConfig {
    Config public_config{};
    ContextType context{nullptr};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config) {
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
    const auto &cfg = config.public_config;
    resource_ = Resource{};
    resource_.initialize(cfg.resource_config);

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

    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{};
    auto context = makePayloadContext();

    typename Core::template Builder<
        typename BufferPayloadPoolTraitsT<ResourceT>::Request,
        typename BufferPayloadPoolTraitsT<ResourceT>::Context>{}
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

  friend struct ContextPayloadPoolTraits;

public:
  void shutdown() {
    if (!core_.isConfigured()) {
      return;
    }
    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{};
    const auto context = makePayloadContext();
    core_.shutdown(request, context);
    resource_ = Resource{};
    context_ = nullptr;
    ops_ = nullptr;
  }

  BufferLease acquire(std::size_t size, std::size_t alignment = 0) {
    core_.ensureConfigured();
    if (size == 0) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "CUDA buffer manager requires size > 0");
    }
    typename BufferPayloadPoolTraitsT<ResourceT>::Request request{};
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

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, ContextType context,
                        SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.context = context;
    internal.ops = ops;
    configure(internal);
  }

  bool isConfiguredForTest() const noexcept { return core_.isConfigured(); }
  std::size_t payloadPoolSizeForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  std::size_t payloadPoolCapacityForTest() const noexcept {
    return core_.payloadPoolCapacityForTest();
  }
  std::size_t controlBlockPoolSizeForTest() const noexcept {
    return core_.controlBlockPoolSizeForTest();
  }
  std::size_t controlBlockPoolCapacityForTest() const noexcept {
    return core_.controlBlockPoolCapacityForTest();
  }
  bool isAliveForTest(BufferHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  typename BufferPayloadPoolTraitsT<ResourceT>::Context makePayloadContext()
      noexcept {
    typename BufferPayloadPoolTraitsT<ResourceT>::Context context{};
    context.context = context_;
    context.ops = ops_;
    context.resource = &resource_;
    return context;
  }

  Resource resource_{};
  ContextType context_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
};

using CudaBufferManager =
    CudaBufferManagerT<
        ::orteaf::internal::execution::cuda::resource::CudaResource>;

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
