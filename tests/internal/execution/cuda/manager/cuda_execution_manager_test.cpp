#include "orteaf/internal/execution/cuda/manager/cuda_execution_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#if ORTEAF_ENABLE_CUDA

namespace cuda_rt = orteaf::internal::execution::cuda::manager;
namespace cuda_platform = orteaf::internal::execution::cuda::platform;
namespace cuda_wrapper = orteaf::internal::execution::cuda::platform::wrapper;

namespace {

class TestCudaSlowOps final : public cuda_platform::CudaSlowOps {
public:
  int getDeviceCount() override { return 1; }

  cuda_wrapper::CudaDevice_t getDevice(std::uint32_t) override {
    return cuda_wrapper::CudaDevice_t{0};
  }

  cuda_wrapper::ComputeCapability getComputeCapability(
      cuda_wrapper::CudaDevice_t) override {
    return cuda_wrapper::ComputeCapability{8, 0};
  }

  std::string getDeviceName(cuda_wrapper::CudaDevice_t) override {
    return "mock-cuda";
  }

  std::string getDeviceVendor(cuda_wrapper::CudaDevice_t) override {
    return "nvidia";
  }

  cuda_wrapper::CudaContext_t getPrimaryContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  cuda_wrapper::CudaContext_t createContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  void setContext(cuda_wrapper::CudaContext_t) override {}

  void releasePrimaryContext(cuda_wrapper::CudaDevice_t) override {}

  void releaseContext(cuda_wrapper::CudaContext_t) override {}

  cuda_wrapper::CudaStream_t createStream() override { return nullptr; }

  void destroyStream(cuda_wrapper::CudaStream_t) override {}

  cuda_wrapper::CudaEvent_t createEvent() override { return nullptr; }

  void destroyEvent(cuda_wrapper::CudaEvent_t) override {}

  cuda_wrapper::CudaModule_t loadModuleFromFile(const char *) override {
    return nullptr;
  }

  cuda_wrapper::CudaModule_t loadModuleFromImage(const void *) override {
    return nullptr;
  }

  cuda_wrapper::CudaFunction_t getFunction(cuda_wrapper::CudaModule_t,
                                           const char *) override {
    return nullptr;
  }

  void unloadModule(cuda_wrapper::CudaModule_t) override {}

private:
  cuda_wrapper::CudaContext_t context_{
      reinterpret_cast<cuda_wrapper::CudaContext_t>(0x1)};
};

cuda_rt::CudaExecutionManager::Config makeConfig(
    cuda_platform::CudaSlowOps *ops) {
  cuda_rt::CudaExecutionManager::Config config{};
  config.slow_ops = ops;

  config.device_config.control_block_capacity = 1;
  config.device_config.control_block_block_size = 1;
  config.device_config.payload_capacity = 1;
  config.device_config.payload_block_size = 1;

  auto &context_cfg = config.device_config.context_config;
  context_cfg.control_block_capacity = 2;
  context_cfg.control_block_block_size = 2;
  context_cfg.payload_capacity = 2;
  context_cfg.payload_block_size = 2;

  context_cfg.stream_config.control_block_capacity = 1;
  context_cfg.stream_config.control_block_block_size = 1;
  context_cfg.stream_config.payload_capacity = 1;
  context_cfg.stream_config.payload_block_size = 1;

  context_cfg.event_config.control_block_capacity = 1;
  context_cfg.event_config.control_block_block_size = 1;
  context_cfg.event_config.payload_capacity = 1;
  context_cfg.event_config.payload_block_size = 1;

  context_cfg.buffer_config.control_block_capacity = 1;
  context_cfg.buffer_config.control_block_block_size = 1;
  context_cfg.buffer_config.payload_capacity = 1;
  context_cfg.buffer_config.payload_block_size = 1;

  context_cfg.module_config.control_block_capacity = 1;
  context_cfg.module_config.control_block_block_size = 1;
  context_cfg.module_config.payload_capacity = 1;
  context_cfg.module_config.payload_block_size = 1;
  context_cfg.module_config.function_config.control_block_capacity = 1;
  context_cfg.module_config.function_config.control_block_block_size = 1;
  context_cfg.module_config.function_config.payload_capacity = 1;
  context_cfg.module_config.function_config.payload_block_size = 1;

  config.kernel_base_config.control_block_capacity = 2;
  config.kernel_base_config.control_block_block_size = 2;
  config.kernel_base_config.payload_capacity = 2;
  config.kernel_base_config.payload_block_size = 2;

  config.kernel_metadata_config.control_block_capacity = 2;
  config.kernel_metadata_config.control_block_block_size = 2;
  config.kernel_metadata_config.payload_capacity = 2;
  config.kernel_metadata_config.payload_block_size = 2;

  return config;
}

} // namespace

class CudaExecutionManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    manager_ = std::make_unique<cuda_rt::CudaExecutionManager>();
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
  }

  std::unique_ptr<cuda_rt::CudaExecutionManager> manager_;
};

TEST_F(CudaExecutionManagerTest, ConfigureWithCustomOpsConfiguresManagers) {
  auto *ops = new TestCudaSlowOps();

  manager_->configure(makeConfig(ops));

  EXPECT_TRUE(manager_->isConfigured());
  EXPECT_EQ(manager_->slowOps(), ops);
  EXPECT_TRUE(manager_->deviceManager().isConfiguredForTest());
  EXPECT_TRUE(manager_->kernelBaseManager().isConfiguredForTest());
  EXPECT_TRUE(manager_->kernelMetadataManager().isConfiguredForTest());
}

TEST_F(CudaExecutionManagerTest, ShutdownClearsAllState) {
  manager_->configure(makeConfig(new TestCudaSlowOps()));
  ASSERT_TRUE(manager_->isConfigured());

  manager_->shutdown();

  EXPECT_FALSE(manager_->isConfigured());
}

#endif // ORTEAF_ENABLE_CUDA
