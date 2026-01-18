#pragma once

#if ORTEAF_ENABLE_CUDA

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/lease/strong_lease.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_stream.h"

namespace orteaf::internal::runtime::cuda::manager {

struct ContextPayloadPoolTraits;

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct StreamPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t;
  using Handle = ::orteaf::internal::execution::cuda::CudaStreamHandle;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    ContextType context{nullptr};
    SlowOps *ops{nullptr};
  };

  static bool create(Payload &payload, const Request &,
                     const Context &context) {
    if (context.ops == nullptr || context.context == nullptr) {
      return false;
    }
    context.ops->setContext(context.context);
    auto stream = context.ops->createStream();
    if (stream == nullptr) {
      return false;
    }
    payload = stream;
    return true;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    if (payload != nullptr && context.ops != nullptr &&
        context.context != nullptr) {
      context.ops->setContext(context.context);
      context.ops->destroyStream(payload);
      payload = nullptr;
    }
  }
};

using StreamPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<StreamPayloadPoolTraits>;

// =============================================================================
// ControlBlock
// =============================================================================

struct StreamControlBlockTag {};

using StreamControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaStreamHandle,
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t,
    StreamPayloadPool>;

// =============================================================================
// Manager Traits
// =============================================================================

struct CudaStreamManagerTraits {
  using PayloadPool = StreamPayloadPool;
  using ControlBlock = StreamControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaStreamHandle;
  static constexpr const char *Name = "CUDA stream manager";
};

// =============================================================================
// CudaStreamManager
// =============================================================================

class CudaStreamManager {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using StreamHandle = ::orteaf::internal::execution::cuda::CudaStreamHandle;
  using StreamType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaStream_t;

  using Core = ::orteaf::internal::base::PoolManager<CudaStreamManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using StreamLease = Core::StrongLeaseType;

private:
  friend StreamLease;

public:
  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaStreamManager() = default;
  CudaStreamManager(const CudaStreamManager &) = delete;
  CudaStreamManager &operator=(const CudaStreamManager &) = delete;
  CudaStreamManager(CudaStreamManager &&) = default;
  CudaStreamManager &operator=(CudaStreamManager &&) = default;
  ~CudaStreamManager() = default;

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

  StreamLease acquire();

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
  bool isAliveForTest(StreamHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  StreamPayloadPoolTraits::Context makePayloadContext() const noexcept;

  ContextType context_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
