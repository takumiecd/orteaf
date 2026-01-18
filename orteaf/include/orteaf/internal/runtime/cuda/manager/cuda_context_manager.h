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
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_context.h"
#include "orteaf/internal/runtime/cuda/manager/cuda_buffer_manager.h"
#include "orteaf/internal/runtime/cuda/manager/cuda_event_manager.h"
#include "orteaf/internal/runtime/cuda/manager/cuda_stream_manager.h"

namespace orteaf::internal::runtime::cuda::manager {

struct DevicePayloadPoolTraits;

// =============================================================================
// Context Resource
// =============================================================================

enum class ContextKind : std::uint8_t { Primary, Owned };

struct CudaContextResource {
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using DeviceType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;

  DeviceType device{};
  ContextType context{nullptr};
  bool is_primary{false};
  CudaBufferManager buffer_manager{};
  CudaStreamManager stream_manager{};
  CudaEventManager event_manager{};

  CudaContextResource() = default;
  CudaContextResource(const CudaContextResource &) = delete;
  CudaContextResource &operator=(const CudaContextResource &) = delete;

  CudaContextResource(CudaContextResource &&other) noexcept { moveFrom(std::move(other)); }

  CudaContextResource &operator=(CudaContextResource &&other) noexcept {
    if (this != &other) {
      reset(nullptr);
      moveFrom(std::move(other));
    }
    return *this;
  }

  ~CudaContextResource() { reset(nullptr); }

  void reset(SlowOps *ops) noexcept {
    buffer_manager.shutdown();
    stream_manager.shutdown();
    event_manager.shutdown();
    if (context != nullptr && ops != nullptr) {
      if (is_primary) {
        ops->releasePrimaryContext(device);
      } else {
        ops->releaseContext(context);
      }
    }
    context = nullptr;
    device = DeviceType{};
    is_primary = false;
  }

private:
  void moveFrom(CudaContextResource &&other) noexcept {
    device = other.device;
    context = other.context;
    is_primary = other.is_primary;
    buffer_manager = std::move(other.buffer_manager);
    stream_manager = std::move(other.stream_manager);
    event_manager = std::move(other.event_manager);
    other.device = DeviceType{};
    other.context = nullptr;
    other.is_primary = false;
  }
};

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct ContextPayloadPoolTraits {
  using Payload = CudaContextResource;
  using Handle = ::orteaf::internal::execution::cuda::CudaContextHandle;
  using DeviceType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t;
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
    ContextKind kind{ContextKind::Primary};
  };

  struct Context {
    DeviceType device{};
    SlowOps *ops{nullptr};
    CudaBufferManager::Config buffer_config{};
    CudaStreamManager::Config stream_config{};
    CudaEventManager::Config event_config{};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr) {
      return false;
    }
    if (!request.handle.isValid()) {
      return false;
    }
    payload.device = context.device;
    payload.is_primary = (request.kind == ContextKind::Primary);
    if (payload.is_primary) {
      payload.context = context.ops->getPrimaryContext(context.device);
    } else {
      payload.context = context.ops->createContext(context.device);
    }
    if (payload.context == nullptr) {
      return false;
    }

    CudaBufferManager::InternalConfig buffer_config{};
    buffer_config.public_config = context.buffer_config;
    buffer_config.context = payload.context;
    buffer_config.ops = context.ops;
    payload.buffer_manager.configure(buffer_config);

    CudaStreamManager::InternalConfig stream_config{};
    stream_config.public_config = context.stream_config;
    stream_config.context = payload.context;
    stream_config.ops = context.ops;
    payload.stream_manager.configure(stream_config);

    CudaEventManager::InternalConfig event_config{};
    event_config.public_config = context.event_config;
    event_config.context = payload.context;
    event_config.ops = context.ops;
    payload.event_manager.configure(event_config);

    return true;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    payload.reset(context.ops);
  }
};

using ContextPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<ContextPayloadPoolTraits>;

// =============================================================================
// ControlBlock
// =============================================================================

struct ContextControlBlockTag {};

using ContextControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cuda::CudaContextHandle, CudaContextResource,
    ContextPayloadPool>;

// =============================================================================
// Manager Traits
// =============================================================================

struct CudaContextManagerTraits {
  using PayloadPool = ContextPayloadPool;
  using ControlBlock = ContextControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cuda::CudaContextHandle;
  static constexpr const char *Name = "CUDA context manager";
};

// =============================================================================
// CudaContextManager
// =============================================================================

class CudaContextManager {
public:
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using DeviceType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaDevice_t;
  using ContextType =
      ::orteaf::internal::execution::cuda::platform::wrapper::CudaContext_t;
  using ContextHandle = ::orteaf::internal::execution::cuda::CudaContextHandle;

  using Core = ::orteaf::internal::base::PoolManager<CudaContextManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using ContextLease = Core::StrongLeaseType;

private:
  friend ContextLease;

public:
  struct Config {
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
    CudaBufferManager::Config buffer_config{};
    CudaStreamManager::Config stream_config{};
    CudaEventManager::Config event_config{};
  };

  CudaContextManager() = default;
  CudaContextManager(const CudaContextManager &) = delete;
  CudaContextManager &operator=(const CudaContextManager &) = delete;
  CudaContextManager(CudaContextManager &&) = default;
  CudaContextManager &operator=(CudaContextManager &&) = default;
  ~CudaContextManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    DeviceType device{};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend struct DevicePayloadPoolTraits;

public:
  void shutdown();

  ContextLease acquirePrimary();
  ContextLease acquireOwned();

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, DeviceType device, SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.device = device;
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
  bool isAliveForTest(ContextHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
#endif

private:
  ContextPayloadPoolTraits::Context makePayloadContext() const noexcept;

  DeviceType device_{};
  SlowOps *ops_{nullptr};
  CudaBufferManager::Config buffer_config_{};
  CudaStreamManager::Config stream_config_{};
  CudaEventManager::Config event_config_{};
  Core core_{};
};

} // namespace orteaf::internal::runtime::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
