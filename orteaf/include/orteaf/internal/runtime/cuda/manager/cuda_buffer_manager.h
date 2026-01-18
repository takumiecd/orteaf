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

namespace orteaf::internal::runtime::cuda::manager {

struct ContextPayloadPoolTraits;

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct BufferPayloadPoolTraits {
  using Payload = ::orteaf::internal::execution::cuda::resource::CudaBuffer;
  using Handle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  using BufferView =
      ::orteaf::internal::execution::cuda::resource::CudaBuffer::BufferView;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using Resource =
      ::orteaf::internal::execution::cuda::resource::CudaResource;
  using AllocFn = BufferView (*)(std::size_t, std::size_t);
  using FreeFn = void (*)(BufferView, std::size_t, std::size_t);

  struct Request {
    std::size_t size{0};
    std::size_t alignment{0};
    Handle handle{Handle::invalid()};
  };

  struct Context {
    ContextType context{nullptr};
    SlowOps *ops{nullptr};
    AllocFn alloc{nullptr};
    FreeFn free{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || context.context == nullptr ||
        context.alloc == nullptr || context.free == nullptr ||
        !request.handle.isValid()) {
      return false;
    }

    if (request.size == 0) {
      payload = Payload{};
      return true;
    }

    context.ops->setContext(context.context);
    auto view = context.alloc(request.size, request.alignment);
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
        context.free != nullptr) {
      context.ops->setContext(context.context);
      context.free(payload.view, request.size, request.alignment);
    }
    payload = Payload{};
  }
};

// =============================================================================
// Payload Pool
// =============================================================================

using BufferPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<BufferPayloadPoolTraits>;

struct BufferManagerCBTag {};

// =============================================================================
// ControlBlock
// =============================================================================

using BufferControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaBufferHandle,
    ::orteaf::internal::execution::cuda::resource::CudaBuffer,
    BufferPayloadPool>;

// =============================================================================
// Manager Traits
// =============================================================================

struct CudaBufferManagerTraits {
  using PayloadPool = BufferPayloadPool;
  using ControlBlock = BufferControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  static constexpr const char *Name = "CUDA buffer manager";
};

// =============================================================================
// CudaBufferManager
// =============================================================================

class CudaBufferManager {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using BufferHandle = ::orteaf::internal::execution::cuda::CudaBufferHandle;
  using BufferView =
      ::orteaf::internal::execution::cuda::resource::CudaBuffer::BufferView;

  using Core = ::orteaf::internal::base::PoolManager<CudaBufferManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using BufferLease = Core::StrongLeaseType;

  struct Config {
    BufferPayloadPoolTraits::AllocFn alloc{nullptr};
    BufferPayloadPoolTraits::FreeFn free{nullptr};
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaBufferManager() = default;
  CudaBufferManager(const CudaBufferManager &) = delete;
  CudaBufferManager &operator=(const CudaBufferManager &) = delete;
  CudaBufferManager(CudaBufferManager &&) = default;
  CudaBufferManager &operator=(CudaBufferManager &&) = default;
  ~CudaBufferManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    ContextType context{nullptr};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend struct ContextPayloadPoolTraits;

public:
  void shutdown();

  BufferLease acquire(std::size_t size, std::size_t alignment = 0);

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
  BufferPayloadPoolTraits::Context makePayloadContext() const noexcept;

  ContextType context_{nullptr};
  SlowOps *ops_{nullptr};
  BufferPayloadPoolTraits::AllocFn alloc_{nullptr};
  BufferPayloadPoolTraits::FreeFn free_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
