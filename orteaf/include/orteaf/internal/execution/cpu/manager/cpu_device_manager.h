#pragma once

#include <cstddef>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/cpu/manager/cpu_buffer_manager.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"

namespace orteaf::internal::execution::cpu::manager {

// Forward declaration
class CpuRuntimeManager;

// =============================================================================
// Device Resource
// =============================================================================

/**
 * @brief Resource structure holding CPU device state and sub-managers.
 *
 * Similar to MpsDeviceResource, this holds the architecture information
 * and any device-specific sub-managers (e.g., buffer manager).
 */
struct CpuDeviceResource {
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;

  ::orteaf::internal::architecture::Architecture arch{
      ::orteaf::internal::architecture::Architecture::CpuGeneric};
  bool is_alive{false};
  CpuBufferManager buffer_manager{};

  CpuDeviceResource() = default;
  CpuDeviceResource(const CpuDeviceResource &) = delete;
  CpuDeviceResource &operator=(const CpuDeviceResource &) = delete;
  CpuDeviceResource(CpuDeviceResource &&other) noexcept;
  CpuDeviceResource &operator=(CpuDeviceResource &&other) noexcept;
  ~CpuDeviceResource();

  void reset(SlowOps *slow_ops) noexcept;

private:
  void moveFrom(CpuDeviceResource &&other) noexcept;
};

// =============================================================================
// Payload Pool Traits
// =============================================================================

struct DevicePayloadPoolTraits {
  using Payload = CpuDeviceResource;
  using Handle = ::orteaf::internal::execution::cpu::CpuDeviceHandle;
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    SlowOps *ops{nullptr};
    CpuBufferManager::Config buffer_config{};
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context);
  static void destroy(Payload &payload, const Request &request,
                      const Context &context);
};

// =============================================================================
// Payload Pool
// =============================================================================

using DevicePayloadPool =
    ::orteaf::internal::base::pool::FixedSlotStore<DevicePayloadPoolTraits>;

// Forward-declare CB tag to avoid circular dependency
struct DeviceManagerCBTag {};

// =============================================================================
// ControlBlock
// =============================================================================

using DeviceControlBlock = ::orteaf::internal::base::StrongControlBlock<
    ::orteaf::internal::execution::cpu::CpuDeviceHandle, CpuDeviceResource,
    DevicePayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct CpuDeviceManagerTraits {
  using PayloadPool = DevicePayloadPool;
  using ControlBlock = DeviceControlBlock;
  using ControlBlockTag = DeviceManagerCBTag;
  using PayloadHandle = ::orteaf::internal::execution::cpu::CpuDeviceHandle;
  static constexpr const char *Name = "CPU device manager";
};

// =============================================================================
// CpuDeviceManager
// =============================================================================

/**
 * @brief CPU device manager using PoolManager pattern.
 *
 * Manages the host CPU device with the same architecture as MpsDeviceManager.
 * Provides DeviceLease for safe resource access with automatic cleanup.
 */
class CpuDeviceManager {
public:
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;
  using DeviceHandle = ::orteaf::internal::execution::cpu::CpuDeviceHandle;

  using Core = ::orteaf::internal::base::PoolManager<CpuDeviceManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using DeviceLease = Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<DeviceHandle,
                                                               DeviceLease>;

  struct Config {
    // PoolManager settings
    std::size_t control_block_capacity{0};
    std::size_t control_block_block_size{0};
    std::size_t control_block_growth_chunk_size{1};
    std::size_t payload_capacity{0};
    std::size_t payload_block_size{0};
    std::size_t payload_growth_chunk_size{1};
    CpuBufferManager::Config buffer_config{};
  };

  CpuDeviceManager() = default;
  CpuDeviceManager(const CpuDeviceManager &) = delete;
  CpuDeviceManager &operator=(const CpuDeviceManager &) = delete;
  CpuDeviceManager(CpuDeviceManager &&) = default;
  CpuDeviceManager &operator=(CpuDeviceManager &&) = default;
  ~CpuDeviceManager() = default;

  // =========================================================================
  // Lifecycle
  // =========================================================================

private:
  struct InternalConfig {
    Config public_config{};
    SlowOps *ops{nullptr};
  };

  void configure(const InternalConfig &config);

  friend class CpuRuntimeManager;

public:

  /**
   * @brief Shutdown the device manager and release all resources.
   */
  void shutdown();

  // =========================================================================
  // Device access
  // =========================================================================

  /**
   * @brief Acquire a lease for the specified device.
   *
   * @param handle Device handle (must be DeviceHandle{0} for CPU)
   * @return DeviceLease for the device
   */
  DeviceLease acquire(DeviceHandle handle);

#if ORTEAF_ENABLE_TEST
  void configureForTest(const Config &config, SlowOps *ops) {
    InternalConfig internal{};
    internal.public_config = config;
    internal.ops = ops;
    configure(internal);
  }

  std::size_t getDeviceCountForTest() const noexcept;
  bool isConfiguredForTest() const noexcept;
  std::size_t payloadPoolSizeForTest() const noexcept;
  std::size_t payloadPoolCapacityForTest() const noexcept;
  bool isAliveForTest(DeviceHandle handle) const noexcept;
  std::size_t controlBlockPoolAvailableForTest() const noexcept;
#endif

private:
  SlowOps *ops_{nullptr};
  Core core_{};
  LifetimeRegistry lifetime_{};
};

} // namespace orteaf::internal::execution::cpu::manager
