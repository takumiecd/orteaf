#pragma once

#include <memory>

#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/manager/cpu_kernel_base_manager.h"
#include "orteaf/internal/execution/cpu/manager/cpu_kernel_metadata_manager.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"

namespace orteaf::internal::execution::cpu::manager {

/**
 * @brief CPU execution manager that provides unified access to CPU managers.
 *
 * Similar to MpsExecutionManager, this class owns the SlowOps instance and
 * manages the lifecycle of CPU managers (device manager, buffer manager, etc.).
 */
class CpuExecutionManager {
  using SlowOps = ::orteaf::internal::execution::cpu::platform::CpuSlowOps;
  using SlowOpsImpl =
      ::orteaf::internal::execution::cpu::platform::CpuSlowOpsImpl;

public:
  // =========================================================================
  // Config
  // =========================================================================

  struct Config {
    /// Custom SlowOps instance (nullptr for default implementation).
    /// If provided, the ExecutionManager takes ownership.
    SlowOps *slow_ops = nullptr;
    /// Device manager configuration
    CpuDeviceManager::Config device_config = {};
    /// Kernel base manager configuration
    CpuKernelBaseManager::Config kernel_base_config = {};
    /// Kernel metadata manager configuration
    CpuKernelMetadataManager::Config kernel_metadata_config = {};
  };

  CpuExecutionManager() = default;
  CpuExecutionManager(const CpuExecutionManager &) = delete;
  CpuExecutionManager &operator=(const CpuExecutionManager &) = delete;
  CpuExecutionManager(CpuExecutionManager &&) = default;
  CpuExecutionManager &operator=(CpuExecutionManager &&) = default;
  ~CpuExecutionManager() = default;

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

  /**
   * @brief Get the kernel base manager.
   */
  CpuKernelBaseManager &kernelBaseManager() noexcept {
    return kernel_base_manager_;
  }
  const CpuKernelBaseManager &kernelBaseManager() const noexcept {
    return kernel_base_manager_;
  }

  /**
   * @brief Get the kernel metadata manager.
   */
  CpuKernelMetadataManager &kernelMetadataManager() noexcept {
    return kernel_metadata_manager_;
  }
  const CpuKernelMetadataManager &kernelMetadataManager() const noexcept {
    return kernel_metadata_manager_;
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  /**
   * @brief Configure the CPU execution manager.
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

    // Configure kernel base manager
    CpuKernelBaseManager::InternalConfig kernel_base_config{};
    kernel_base_config.public_config = config.kernel_base_config;
    kernel_base_manager_.configure(kernel_base_config);

    // Configure kernel metadata manager
    CpuKernelMetadataManager::InternalConfig kernel_metadata_config{};
    kernel_metadata_config.public_config = config.kernel_metadata_config;
    kernel_metadata_manager_.configure(kernel_metadata_config);
  }

  /**
   * @brief Shutdown the CPU execution manager and release all resources.
   */
  void shutdown() {
    kernel_metadata_manager_.shutdown();
    kernel_base_manager_.shutdown();
    device_manager_.shutdown();
    slow_ops_.reset();
  }

  /**
   * @brief Check if the execution manager is configured.
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
  CpuKernelBaseManager kernel_base_manager_{};
  CpuKernelMetadataManager kernel_metadata_manager_{};
  std::unique_ptr<SlowOps> slow_ops_{};
};

} // namespace orteaf::internal::execution::cpu::manager
