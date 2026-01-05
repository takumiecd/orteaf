#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_queue.h"
#include "orteaf/internal/execution/mps/resource/mps_command_queue_resource.h"

namespace orteaf::internal::execution::mps::manager {

// =============================================================================
// Payload Pool
// =============================================================================

struct CommandQueuePayloadPoolTraits {
  using Payload =
      ::orteaf::internal::execution::mps::resource::MpsCommandQueueResource;
  using Handle = ::orteaf::internal::base::CommandQueueHandle;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;
  using FenceManager = ::orteaf::internal::execution::mps::manager::
      MpsFenceManager;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    DeviceType device{nullptr};
    SlowOps *ops{nullptr};
    FenceManager *fence_manager{nullptr};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || context.device == nullptr) {
      return false;
    }
    auto queue = context.ops->createCommandQueue(context.device);
    if (queue == nullptr) {
      return false;
    }
    payload.setQueue(queue);
    auto &lifetime = payload.lifetime();
    if (!lifetime.setFenceManager(context.fence_manager)) {
      context.ops->destroyCommandQueue(queue);
      payload.setQueue(nullptr);
      return false;
    }
    if (!lifetime.setCommandQueueHandle(request.handle)) {
      context.ops->destroyCommandQueue(queue);
      payload.setQueue(nullptr);
      return false;
    }
    return true;
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    if (!payload.hasQueue() || context.ops == nullptr) {
      return;
    }
    if (!payload.lifetime().empty()) {
      payload.lifetime().waitUntilReady();
    }
    context.ops->destroyCommandQueue(payload.queue());
    payload.setQueue(nullptr);
  }
};

using CommandQueuePayloadPool =
    ::orteaf::internal::base::pool::SlotPool<CommandQueuePayloadPoolTraits>;

// =============================================================================
// ControlBlock (StrongControlBlock for owning references)
// =============================================================================

struct CommandQueueControlBlockTag {};

using CommandQueueControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::base::CommandQueueHandle,
    ::orteaf::internal::execution::mps::resource::MpsCommandQueueResource,
    CommandQueuePayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct MpsCommandQueueManagerTraits {
  using PayloadPool = CommandQueuePayloadPool;
  using ControlBlock = CommandQueueControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::base::CommandQueueHandle;
  static constexpr const char *Name = "MPS command queue manager";
};

// =============================================================================
// MpsCommandQueueManager
// =============================================================================

class MpsCommandQueueManager {
public:
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;
  using CommandQueueHandle = ::orteaf::internal::base::CommandQueueHandle;
  using CommandQueueType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandQueue_t;

  using Core =
      ::orteaf::internal::base::PoolManager<MpsCommandQueueManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using CommandQueueLease = Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<
          CommandQueueHandle, CommandQueueLease>;

private:
  friend CommandQueueLease;

public:
  struct Config {
    DeviceType device{nullptr};
    SlowOps *ops{nullptr};
    Core::Config pool{};
    ::orteaf::internal::execution::mps::manager::MpsFenceManager
        *fence_manager{nullptr};
  };

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  MpsCommandQueueManager() = default;
  MpsCommandQueueManager(const MpsCommandQueueManager &) = delete;
  MpsCommandQueueManager &operator=(const MpsCommandQueueManager &) = delete;
  MpsCommandQueueManager(MpsCommandQueueManager &&) = default;
  MpsCommandQueueManager &operator=(MpsCommandQueueManager &&) = default;
  ~MpsCommandQueueManager() = default;

  void configure(const Config &config);
  void shutdown();

  // ===========================================================================
  // Acquire API
  // ===========================================================================

  /// @brief Acquire a new command queue (creates queue + control block)
  CommandQueueLease acquire();

  /// @brief Acquire a strong reference to an existing queue by handle
  CommandQueueLease acquire(CommandQueueHandle handle);

  template <typename FastOps =
                ::orteaf::internal::execution::mps::platform::MpsFastOps>
  bool releaseReadyAndMaybeRelease(CommandQueueLease &lease) {
    if (!lease) {
      return false;
    }
    const auto handle = lease.payloadHandle();
    auto *payload = lease.payloadPtr();
    if (payload == nullptr) {
      if (handle.isValid()) {
        lifetime_.release(handle);
      }
      lease.release();
      return true;
    }
    auto &fence_lifetime = payload->lifetime();
    fence_lifetime.releaseReady<FastOps>();
    if (!fence_lifetime.empty()) {
      return false;
    }
    if (handle.isValid()) {
      lifetime_.release(handle);
    }
    lease.release();
    return true;
  }

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
  bool isAliveForTest(CommandQueueHandle handle) const noexcept {
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
  Core core_{};
  DeviceType device_{nullptr};
  SlowOps *ops_{nullptr};
  ::orteaf::internal::execution::mps::manager::MpsFenceManager
      *fence_manager_{nullptr};
  LifetimeRegistry lifetime_{};
};

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
