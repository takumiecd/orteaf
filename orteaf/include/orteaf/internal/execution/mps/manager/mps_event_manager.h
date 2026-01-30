#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/lease/strong_lease.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_event.h"

namespace orteaf::internal::execution::mps::manager {

struct DevicePayloadPoolTraits;

// =============================================================================
// Payload
// =============================================================================

struct MpsEventPayload {
  using EventType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsEvent_t;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;

  struct InitConfig {
    DeviceType device{nullptr};
    SlowOps *ops{nullptr};
  };

  MpsEventPayload() = default;
  MpsEventPayload(const MpsEventPayload &) = delete;
  MpsEventPayload &operator=(const MpsEventPayload &) = delete;
  MpsEventPayload(MpsEventPayload &&) = default;
  MpsEventPayload &operator=(MpsEventPayload &&) = default;
  ~MpsEventPayload() = default;

  bool initialize(const InitConfig &config) {
    if (config.ops == nullptr || config.device == nullptr) {
      return false;
    }
    auto event = config.ops->createEvent(config.device);
    if (event == nullptr) {
      return false;
    }
    event_ = event;
    return true;
  }

  void reset(SlowOps *ops) noexcept {
    if (event_ != nullptr && ops != nullptr) {
      ops->destroyEvent(event_);
      event_ = nullptr;
    }
  }

  EventType event() const noexcept { return event_; }
  bool hasEvent() const noexcept { return event_ != nullptr; }

private:
  EventType event_{nullptr};
};

// =============================================================================
// Payload Pool
// =============================================================================

struct EventPayloadPoolTraits {
  using Payload = MpsEventPayload;
  using Handle = ::orteaf::internal::execution::mps::MpsEventHandle;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    DeviceType device{nullptr};
    SlowOps *ops{nullptr};
  };

  static bool create(Payload &payload, const Request &,
                     const Context &context) {
    MpsEventPayload::InitConfig init{};
    init.device = context.device;
    init.ops = context.ops;
    return payload.initialize(init);
  }

  static void destroy(Payload &payload, const Request &,
                      const Context &context) {
    payload.reset(context.ops);
  }
};

using EventPayloadPool =
    ::orteaf::internal::base::pool::SlotPool<EventPayloadPoolTraits>;

// =============================================================================
// ControlBlock (using default pool traits via PoolManager)
// =============================================================================

struct EventControlBlockTag {};

using EventControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::mps::MpsEventHandle,
    MpsEventPayload,
    EventPayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct MpsEventManagerTraits {
  using PayloadPool = EventPayloadPool;
  using ControlBlock = EventControlBlock;
  struct ControlBlockTag {};
  using PayloadHandle = ::orteaf::internal::execution::mps::MpsEventHandle;
  static constexpr const char *Name = "MPS event manager";
};

// =============================================================================
// MpsEventManager
// =============================================================================

class MpsEventManager {
public:
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOps;
  using DeviceType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsDevice_t;
  using EventHandle = ::orteaf::internal::execution::mps::MpsEventHandle;
  using EventType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsEvent_t;

  using Core = ::orteaf::internal::base::PoolManager<MpsEventManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using EventLease = Core::StrongLeaseType;

public:
  struct Config {
    // PoolManager settings
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
  };

  MpsEventManager() = default;
  MpsEventManager(const MpsEventManager &) = delete;
  MpsEventManager &operator=(const MpsEventManager &) = delete;
  MpsEventManager(MpsEventManager &&) = default;
  MpsEventManager &operator=(MpsEventManager &&) = default;
  ~MpsEventManager() = default;

private:
  struct InternalConfig {
    Config public_config{};
    DeviceType device{nullptr};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend struct DevicePayloadPoolTraits;
  friend struct MpsDevicePayload;

public:
  void shutdown();

  EventLease acquire();

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, DeviceType device,
                        SlowOps *ops) {
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
  bool isAliveForTest(EventHandle handle) const noexcept {
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
  EventPayloadPoolTraits::Context makePayloadContext() const noexcept;

  DeviceType device_{nullptr};
  SlowOps *ops_{nullptr};
  Core core_{};
};

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
