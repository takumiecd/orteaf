#pragma once

#include <cstddef>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"

namespace orteaf::internal::execution::cpu::manager {

// Forward declaration
class CpuBufferManager;

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
  using Handle = ::orteaf::internal::base::DeviceHandle;
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;

  struct Request {
    Handle handle{Handle::invalid()};
  };

  struct Context {
    SlowOps *ops{nullptr};
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
    ::orteaf::internal::base::DeviceHandle, CpuDeviceResource,
    DevicePayloadPool>;

// =============================================================================
// Manager Traits for PoolManager
// =============================================================================

struct CpuDeviceManagerTraits {
  using PayloadPool = DevicePayloadPool;
  using ControlBlock = DeviceControlBlock;
  using ControlBlockTag = DeviceManagerCBTag;
  using PayloadHandle = ::orteaf::internal::base::DeviceHandle;
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
  using DeviceHandle = ::orteaf::internal::base::DeviceHandle;

  using Core = ::orteaf::internal::base::PoolManager<CpuDeviceManagerTraits>;
  using ControlBlock = Core::ControlBlock;
  using ControlBlockHandle = Core::ControlBlockHandle;
  using ControlBlockPool = Core::ControlBlockPool;

  using DeviceLease = Core::StrongLeaseType;
  using LifetimeRegistry =
      ::orteaf::internal::base::manager::LeaseLifetimeRegistry<DeviceHandle,
                                                               DeviceLease>;

  struct Config {
    SlowOps *ops{nullptr};
    Core::Config pool{};
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

  /**
   * @brief Configure the device manager.
   *
   * @param config Configuration including SlowOps and pool settings
   */
  void configure(const Config &config);

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

  /**
   * @brief Release a device lease.
   *
   * @param lease Lease to release
   */
  void release(DeviceLease &lease) noexcept;

  /**
   * @brief Get the architecture for the specified device.
   *
   * @param handle Device handle
   * @return Architecture enum value
   */
  ::orteaf::internal::architecture::Architecture getArch(DeviceHandle handle);

  /**
   * @brief Get the number of CPU devices.
   *
   * @return Always 1 for CPU
   */
  std::size_t getDeviceCount() const noexcept;

  /**
   * @brief Check if a device handle is alive.
   *
   * @param handle Device handle to check
   * @return true if device is initialized and alive
   */
  bool isAlive(DeviceHandle handle) const noexcept;

#if ORTEAF_ENABLE_TEST
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
