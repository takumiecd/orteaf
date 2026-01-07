#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/strong.h"
#include "orteaf/internal/base/manager/lease_lifetime_registry.h"
#include "orteaf/internal/base/manager/pool_manager.h"
#include "orteaf/internal/base/pool/fixed_slot_store.h"
#include "orteaf/internal/diagnostics/error/error.h"
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
  // CpuBufferManager will be added in Phase 3

  CpuDeviceResource() = default;
  CpuDeviceResource(const CpuDeviceResource &) = delete;
  CpuDeviceResource &operator=(const CpuDeviceResource &) = delete;

  CpuDeviceResource(CpuDeviceResource &&other) noexcept {
    moveFrom(std::move(other));
  }

  CpuDeviceResource &operator=(CpuDeviceResource &&other) noexcept {
    if (this != &other) {
      reset(nullptr);
      moveFrom(std::move(other));
    }
    return *this;
  }

  ~CpuDeviceResource() { reset(nullptr); }

  void reset([[maybe_unused]] SlowOps *slow_ops) noexcept {
    // Shutdown sub-managers here when added
    arch = ::orteaf::internal::architecture::Architecture::CpuGeneric;
    is_alive = false;
  }

private:
  void moveFrom(CpuDeviceResource &&other) noexcept {
    arch = other.arch;
    is_alive = other.is_alive;
    other.arch = ::orteaf::internal::architecture::Architecture::CpuGeneric;
    other.is_alive = false;
  }
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
    // Config for sub-managers can be added here
  };

  static bool create(Payload &payload, const Request &request,
                     const Context &context) {
    if (context.ops == nullptr || !request.handle.isValid()) {
      return false;
    }

    // CPU has only one device with index 0
    if (request.handle.index != 0) {
      return false;
    }

    payload.arch = context.ops->detectArchitecture(request.handle);
    payload.is_alive = true;

    // Initialize sub-managers here when added
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
    // Sub-manager configs can be added here
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
  void configure(const Config &config) {
    shutdown();

    if (config.ops == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "CPU device manager requires valid ops");
    }

    ops_ = config.ops;

    // Setup pool configuration
    auto pool_config = config.pool;
    // CPU has exactly 1 device
    pool_config.payload_capacity = 1;
    pool_config.payload_block_size = 1;
    if (pool_config.control_block_capacity == 0) {
      pool_config.control_block_capacity = 4;
    }
    if (pool_config.control_block_block_size == 0) {
      pool_config.control_block_block_size = 4;
    }

    DevicePayloadPoolTraits::Request request{};
    request.handle = DeviceHandle{0};

    DevicePayloadPoolTraits::Context context{};
    context.ops = ops_;

    core_.configure(pool_config, request, context);
    core_.createAllPayloads(request, context);
  }

  /**
   * @brief Shutdown the device manager and release all resources.
   */
  void shutdown() {
    lifetime_.clear();

    DevicePayloadPoolTraits::Request request{};
    DevicePayloadPoolTraits::Context context{};
    context.ops = ops_;

    core_.shutdown(request, context);
    ops_ = nullptr;
  }

  // =========================================================================
  // Device access
  // =========================================================================

  /**
   * @brief Acquire a lease for the specified device.
   *
   * @param handle Device handle (must be DeviceHandle{0} for CPU)
   * @return DeviceLease for the device
   */
  DeviceLease acquire(DeviceHandle handle) {
    core_.ensureConfigured();

    if (!handle.isValid() || handle.index != 0) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "invalid CPU device handle");
    }

    // Check if we already have an active lease
    if (lifetime_.has(handle)) {
      return lifetime_.get(handle);
    }

    // Create new lease
    auto lease = core_.acquireStrongLease(handle);
    if (lease) {
      lifetime_.set(DeviceLease{lease});
    }
    return lease;
  }

  /**
   * @brief Release a device lease.
   *
   * @param lease Lease to release
   */
  void release(DeviceLease &lease) noexcept { lease.release(); }

  /**
   * @brief Get the architecture for the specified device.
   *
   * @param handle Device handle
   * @return Architecture enum value
   */
  ::orteaf::internal::architecture::Architecture getArch(DeviceHandle handle) {
    core_.ensureConfigured();

    if (!handle.isValid() || handle.index != 0) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "invalid CPU device handle");
    }

    // Get lease from lifetime registry
    if (!lifetime_.has(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CPU device not acquired");
    }

    auto lease = lifetime_.get(handle);
    if (!lease) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CPU device lease is invalid");
    }

    return lease.payloadPtr()->arch;
  }

  /**
   * @brief Get the number of CPU devices.
   *
   * @return Always 1 for CPU
   */
  std::size_t getDeviceCount() const noexcept {
    return core_.isConfigured() ? 1u : 0u;
  }

  /**
   * @brief Check if a device handle is alive.
   *
   * @param handle Device handle to check
   * @return true if device is initialized and alive
   */
  bool isAlive(DeviceHandle handle) const noexcept {
    if (!core_.isConfigured() || !handle.isValid() || handle.index != 0) {
      return false;
    }
    return core_.isAlive(handle);
  }

#if ORTEAF_ENABLE_TEST
  std::size_t getDeviceCountForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  bool isConfiguredForTest() const noexcept { return core_.isConfigured(); }
  std::size_t payloadPoolSizeForTest() const noexcept {
    return core_.payloadPoolSizeForTest();
  }
  std::size_t payloadPoolCapacityForTest() const noexcept {
    return core_.payloadPoolCapacityForTest();
  }
  bool isAliveForTest(DeviceHandle handle) const noexcept {
    return core_.isAlive(handle);
  }
  std::size_t controlBlockPoolAvailableForTest() const noexcept {
    return core_.controlBlockPoolAvailableForTest();
  }
#endif

private:
  SlowOps *ops_{nullptr};
  Core core_{};
  LifetimeRegistry lifetime_{};
};

} // namespace orteaf::internal::execution::cpu::manager
