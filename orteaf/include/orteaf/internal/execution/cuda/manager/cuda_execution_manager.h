#pragma once

#if ORTEAF_ENABLE_CUDA

#include <memory>

#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/manager/cuda_device_manager.h"
#include "orteaf/internal/execution/cuda/manager/cuda_kernel_base_manager.h"
#include "orteaf/internal/execution/cuda/manager/cuda_kernel_metadata_manager.h"

namespace orteaf::internal::execution::cuda::manager {

class CudaExecutionManager {
  using SlowOps = ::orteaf::internal::execution::cuda::platform::CudaSlowOps;
  using SlowOpsImpl =
      ::orteaf::internal::execution::cuda::platform::CudaSlowOpsImpl;

public:
  struct Config {
    SlowOps *slow_ops = nullptr;
    CudaDeviceManager::Config device_config = {};
    CudaKernelBaseManager::Config kernel_base_config = {};
    CudaKernelMetadataManager::Config kernel_metadata_config = {};
  };

  CudaExecutionManager() = default;
  CudaExecutionManager(const CudaExecutionManager &) = delete;
  CudaExecutionManager &operator=(const CudaExecutionManager &) = delete;
  CudaExecutionManager(CudaExecutionManager &&) = default;
  CudaExecutionManager &operator=(CudaExecutionManager &&) = default;
  ~CudaExecutionManager() = default;

  CudaDeviceManager &deviceManager() noexcept { return device_manager_; }
  const CudaDeviceManager &deviceManager() const noexcept {
    return device_manager_;
  }

  CudaKernelBaseManager &kernelBaseManager() noexcept {
    return kernel_base_manager_;
  }
  const CudaKernelBaseManager &kernelBaseManager() const noexcept {
    return kernel_base_manager_;
  }

  CudaKernelMetadataManager &kernelMetadataManager() noexcept {
    return kernel_metadata_manager_;
  }
  const CudaKernelMetadataManager &kernelMetadataManager() const noexcept {
    return kernel_metadata_manager_;
  }

  SlowOps *slowOps() noexcept { return slow_ops_.get(); }
  const SlowOps *slowOps() const noexcept { return slow_ops_.get(); }

  void configure() { configure(Config{}); }

  void configure(const Config &config) {
    if (config.slow_ops) {
      slow_ops_.reset(config.slow_ops);
    } else if (!slow_ops_) {
      slow_ops_ = std::make_unique<SlowOpsImpl>();
    }

    CudaDeviceManager::InternalConfig device_config{};
    device_config.public_config = config.device_config;
    device_config.ops = slow_ops_.get();
    device_manager_.configure(device_config);

    CudaKernelBaseManager::InternalConfig kernel_base_config{};
    kernel_base_config.public_config = config.kernel_base_config;
    kernel_base_manager_.configure(kernel_base_config);

    CudaKernelMetadataManager::InternalConfig kernel_metadata_config{};
    kernel_metadata_config.public_config = config.kernel_metadata_config;
    kernel_metadata_manager_.configure(kernel_metadata_config);
  }

  void shutdown() {
    kernel_metadata_manager_.shutdown();
    kernel_base_manager_.shutdown();
    device_manager_.shutdown();
    slow_ops_.reset();
  }

  bool isConfigured() const noexcept {
#if ORTEAF_ENABLE_TEST
    return slow_ops_ != nullptr && device_manager_.isConfiguredForTest() &&
           kernel_base_manager_.isConfiguredForTest() &&
           kernel_metadata_manager_.isConfiguredForTest();
#else
    return slow_ops_ != nullptr;
#endif
  }

private:
  CudaDeviceManager device_manager_{};
  CudaKernelBaseManager kernel_base_manager_{};
  CudaKernelMetadataManager kernel_metadata_manager_{};
  std::unique_ptr<SlowOps> slow_ops_{};
};

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
