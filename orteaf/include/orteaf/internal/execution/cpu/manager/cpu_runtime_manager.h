#pragma once

#include <memory>

#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"

namespace orteaf::internal::execution::cpu::manager {

/**
 * @brief CPU runtime manager that provides unified access to CPU managers.
 *
 * Similar to MpsRuntimeManager, this class owns the SlowOps instance and
 * manages the lifecycle of CPU managers (device manager, buffer manager, etc.).
 */
class CpuRuntimeManager {
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;
  using SlowOpsImpl =
      ::orteaf::internal::execution::cpu::platform::CpuSlowOpsImpl;

public:
  // =========================================================================
  // Config
  // =========================================================================

  struct Config {
    /// Custom SlowOps instance (nullptr for default implementation).
    /// If provided, the RuntimeManager takes ownership.
    SlowOps *slow_ops = nullptr;
    /// Device manager configuration
    CpuDeviceManager::Config device_config = {};
  };

  CpuRuntimeManager() = default;
  CpuRuntimeManager(const CpuRuntimeManager &) = delete;
  CpuRuntimeManager &operator=(const CpuRuntimeManager &) = delete;
  CpuRuntimeManager(CpuRuntimeManager &&) = default;
  CpuRuntimeManager &operator=(CpuRuntimeManager &&) = default;
  ~CpuRuntimeManager() = default;

  // =========================================================================
  // Manager accessors
  // =========================================================================

  /**
   * @brief Get the device manager.
   */
  CpuDeviceManager &deviceManager() noexcept { return device_manager_; }
  const CpuDeviceManager &deviceManager() const noexcept {
    return device_manager_;
  }

  /**
   * @brief Get the SlowOps instance.
   */
  SlowOps *slowOps() noexcept { return slow_ops_.get(); }
  const SlowOps *slowOps() const noexcept { return slow_ops_.get(); }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  /**
   * @brief Configure the CPU runtime with default settings.
   */
  void configure() { configure(Config{}); }

  /**
   * @brief Configure the CPU runtime.
   *
   * @param config Configuration including SlowOps and sub-manager settings
   */
  void configure(const Config &config) {
    if (config.slow_ops) {
      slow_ops_.reset(config.slow_ops);
    } else if (!slow_ops_) {
      slow_ops_ = std::make_unique<SlowOpsImpl>();
    }

    // Configure device manager
    CpuDeviceManager::InternalConfig device_config{};
    device_config.public_config = config.device_config;
    device_config.ops = slow_ops_.get();
    device_manager_.configure(device_config);
  }

  /**
   * @brief Shutdown the CPU runtime and release all resources.
   */
  void shutdown() {
    device_manager_.shutdown();
    slow_ops_.reset();
  }

  /**
   * @brief Check if the runtime is configured.
   */
  bool isConfigured() const noexcept {
#if ORTEAF_ENABLE_TEST
    return slow_ops_ != nullptr && device_manager_.isConfiguredForTest();
#else
    return slow_ops_ != nullptr;
#endif
  }

private:
  CpuDeviceManager device_manager_{};
  std::unique_ptr<SlowOps> slow_ops_{};
};

} // namespace orteaf::internal::execution::cpu::manager
