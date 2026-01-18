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
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_event.h"

namespace orteaf::internal::runtime::cuda::manager {

struct ContextPayloadPoolTraits;

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct EventPayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t;
  using Handle = ::orteaf::internal::execution::cuda::CudaEventHandle;
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
    auto event = context.ops->createEvent();
    if (event == nullptr) {
      return false;
    }
    payload = event;
    return true;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    if (payload != nullptr && context.ops != nullptr &&
        context.context != nullptr) {
      context.ops->setContext(context.context);
      context.ops->destroyEvent(payload);
      payload = nullptr;
    }
  }
};

using EventPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<EventPayloadPoolTraits>;

// =============================================================================
// ControlBlock
// =============================================================================

struct EventControlBlockTag {};

using EventControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaEventHandle,
    ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t,
    EventPayloadPool>;

// =============================================================================
// Manager Traits
// =============================================================================

struct CudaEventManagerTraits {
  using PayloadPool = EventPayloadPool;
  using ControlBlock = EventControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaEventHandle;
  static constexpr const char *Name = "CUDA event manager";
};

// =============================================================================
// CudaEventManager
// =============================================================================

class CudaEventManager {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using EventHandle = ::orteaf::internal::execution::cuda::CudaEventHandle;
  using EventType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaEvent_t;

  using Core = ::orteaf::internal::base::PoolManager<CudaEventManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using EventLease = Core::StrongLeaseType;

private:
  friend EventLease;

public:
  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CudaEventManager() = default;
  CudaEventManager(const CudaEventManager &) = delete;
  CudaEventManager &operator=(const CudaEventManager &) = delete;
  CudaEventManager(CudaEventManager &&) = default;
  CudaEventManager &operator=(CudaEventManager &&) = default;
  ~CudaEventManager() = default;

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

  EventLease acquire();

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
  bool isAliveForTest(EventHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  EventPayloadPoolTraits::Context makePayloadContext() const noexcept;

  ContextType context_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
