#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"
#include "orteaf/internal/execution/cpu/resource/cpu_buffer_view.h"

namespace orteaf::internal::execution::cpu::manager {

class CpuRuntimeManager;

// =============================================================================
// Buffer Resource
// =============================================================================

/**
 * @brief Resource structure holding CPU buffer state.
 *
 * Represents an allocated CPU memory buffer with size and alignment info.
 */
struct CpuBufferResource {
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;

  void *data{nullptr};
  std::size_t size{0};
  std::size_t alignment{0};

  CpuBufferResource() = default;
  CpuBufferResource(const CpuBufferResource &) = delete;
  CpuBufferResource &operator=(const CpuBufferResource &) = delete;

  CpuBufferResource(CpuBufferResource &&other) noexcept {
    moveFrom(std::move(other));
  }

  CpuBufferResource &operator=(CpuBufferResource &&other) noexcept {
    if (this != &other) {
      reset(nullptr);
      moveFrom(std::move(other));
    }
    return *this;
  }

  ~CpuBufferResource() { reset(nullptr); }

  /**
   * @brief Reset and deallocate the buffer.
   *
   * @param slow_ops SlowOps for deallocation (can be nullptr to skip dealloc)
   */
  void reset(SlowOps *slow_ops) noexcept {
    if (data != nullptr && slow_ops != nullptr) {
      slow_ops->deallocBuffer(data, size);
    }
    data = nullptr;
    size = 0;
    alignment = 0;
  }

  /**
   * @brief Create a buffer view for this resource.
   *
   * @return CpuBufferView representing the full buffer
   */
  ::orteaf::internal::execution::cpu::resource::CpuBufferView view() const {
    return ::orteaf::internal::execution::cpu::resource::CpuBufferView{data, 0,
                                                                       size};
  }

  /**
   * @brief Check if buffer is empty.
   */
  bool empty() const noexcept { return data == nullptr; }

private:
  void moveFrom(CpuBufferResource &&other) noexcept {
    data = other.data;
    size = other.size;
    alignment = other.alignment;
    other.data = nullptr;
    other.size = 0;
    other.alignment = 0;
  }
};

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct BufferPayloadPoolTraits {
  using Payload = CpuBufferResource;
  using Handle = ::orteaf::internal::execution::cpu::CpuBufferHandle;
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;

  struct Request {
    std::size_t size{0};
    std::size_t alignment{0};
  };

  struct Context {
    SlowOps *ops{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || request.size == 0) {
      return false;
    }

    void *ptr = context.ops->allocBuffer(request.size, request.alignment);
    if (ptr == nullptr) {
      return false;
    }

    payload.data = ptr;
    payload.size = request.size;
    payload.alignment = request.alignment;
    return true;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    payload.reset(context.ops);
  }
};

// =============================================================================
// Payload Pool
// =============================================================================

using BufferPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<BufferPayloadPoolTraits>;

// Forward-declare CB tag
struct BufferManagerCBTag {};

// =============================================================================
// ControlBlock
// =============================================================================

using BufferControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cpu::CpuBufferHandle, CpuBufferResource,
    BufferPayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct CpuBufferManagerTraits {
  using PayloadPool = BufferPayloadPool;
  using ControlBlock = BufferControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::cpu::CpuBufferHandle;
  static constexpr const char *Name = "CPU buffer manager";
};

// =============================================================================
// CpuBufferManager
// =============================================================================

/**
 * @brief CPU buffer manager using PoolManager pattern.
 *
 * Manages CPU memory buffers with pooled allocation.
 * Provides BufferLease for safe resource access with automatic cleanup.
 */
class CpuBufferManager {
public:
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;
  using BufferHandle = ::orteaf::internal::execution::cpu::CpuBufferHandle;
  using BufferView =
      ::orteaf::internal::execution::cpu::resource::CpuBufferView;

  using Core = ::orteaf::internal::base::PoolManager<CpuBufferManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using BufferLease = Core::StrongLeaseType;

  struct Config {
    // PoolManager settings
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  CpuBufferManager() = default;
  CpuBufferManager(const CpuBufferManager &) = delete;
  CpuBufferManager &operator=(const CpuBufferManager &) = delete;
  CpuBufferManager(CpuBufferManager &&) = default;
  CpuBufferManager &operator=(CpuBufferManager &&) = default;
  ~CpuBufferManager() = default;

  // =========================================================================
  // Lifecycle
  // =========================================================================

private:
  struct InternalConfig {
    Config public_config{};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config) {
    ops_ = config.ops;
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
    context.ops = ops_;

    Core::Builder<BufferPayloadPoolTraits::Request,
                  BufferPayloadPoolTraits::Context>{}
        .withControlBlockCapacity(control_block_capacity)
        .withControlBlockBlockSize(control_block_block_size)
        .withControlBlockGrowthChunkSize(
            cfg.control_block_growth_chunk_size)
        .withPayloadCapacity(payload_capacity)
        .withPayloadBlockSize(payload_block_size)
        .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
        .withRequest(request)
        .withContext(context)
        .configure(core_);
  }

  friend class CpuRuntimeManager;

public:
#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.ops = ops;
    configure(internal);
  }
#endif

  /**
   * @brief Shutdown the buffer manager and release all resources.
   */
  void shutdown() {
    BufferPayloadPoolTraits::Request request{};
    BufferPayloadPoolTraits::Context context{};
    context.ops = ops_;

    core_.shutdown(request, context);
    ops_ = nullptr;
  }

  // =========================================================================
  // Buffer operations
  // =========================================================================

  /**
   * @brief Acquire a buffer with specified size and alignment.
   *
   * @param size Size in bytes
   * @param alignment Alignment requirement (0 for default)
   * @return BufferLease for the allocated buffer
   */
  BufferLease acquire(std::size_t size, std::size_t alignment = 0) {
    core_.ensureConfigured();

    if (size == 0) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "buffer size cannot be 0");
    }

    BufferPayloadPoolTraits::Request request{};
    request.size = size;
    request.alignment = alignment;

    BufferPayloadPoolTraits::Context context{};
    context.ops = ops_;

    auto payload_handle = core_.acquirePayloadOrGrowAndCreate(request, context);
    if (!payload_handle.isValid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
          "CPU buffer manager has no available slots");
    }
    return core_.acquireStrongLease(payload_handle);
  }

  /**
   * @brief Release a buffer lease.
   *
   * @param lease Lease to release
   */
  void release(BufferLease &lease) noexcept { lease.release(); }

#if ORTEAF_ENABLE_TEST
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
  std::size_t payloadGrowthChunkSizeForTest() const noexcept {
    return core_.payloadGrowthChunkSize();
  }
  std::size_t controlBlockGrowthChunkSizeForTest() const noexcept {
    return core_.controlBlockGrowthChunkSize();
  }
#endif

private:
  SlowOps *ops_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::execution::cpu::manager
