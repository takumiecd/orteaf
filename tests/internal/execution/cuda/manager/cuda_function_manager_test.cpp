#include "orteaf/internal/execution/cuda/manager/cuda_function_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "tests/internal/testing/error_assert.h"

#if ORTEAF_ENABLE_CUDA

namespace cuda_rt = orteaf::internal::execution::cuda::manager;
namespace cuda_platform = orteaf::internal::execution::cuda::platform;
namespace cuda_wrapper = orteaf::internal::execution::cuda::platform::wrapper;

namespace {

struct TestFunctionHooks {
  int get_function_calls{0};
  std::string last_kernel_name{};
  cuda_wrapper::CudaFunction_t function{
      reinterpret_cast<cuda_wrapper::CudaFunction_t>(0x2)};
};

TestFunctionHooks *g_hooks = nullptr;

class TestCudaSlowOps final : public cuda_platform::CudaSlowOps {
public:
  int getDeviceCount() override { return 1; }

  cuda_wrapper::CudaDevice_t getDevice(std::uint32_t) override {
    return cuda_wrapper::CudaDevice_t{0};
  }

  cuda_wrapper::ComputeCapability getComputeCapability(
      cuda_wrapper::CudaDevice_t) override {
    return cuda_wrapper::ComputeCapability{0, 0};
  }

  std::string getDeviceName(cuda_wrapper::CudaDevice_t) override {
    return "mock-cuda";
  }

  std::string getDeviceVendor(cuda_wrapper::CudaDevice_t) override {
    return "mock";
  }

  cuda_wrapper::CudaContext_t getPrimaryContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  cuda_wrapper::CudaContext_t createContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  void setContext(cuda_wrapper::CudaContext_t context) override {
    last_context_ = context;
  }

  void releasePrimaryContext(cuda_wrapper::CudaDevice_t) override {}

  void releaseContext(cuda_wrapper::CudaContext_t) override {}

  cuda_wrapper::CudaStream_t createStream() override { return nullptr; }

  void destroyStream(cuda_wrapper::CudaStream_t) override {}

  cuda_wrapper::CudaEvent_t createEvent() override { return nullptr; }

  void destroyEvent(cuda_wrapper::CudaEvent_t) override {}

  void setContextForTest(cuda_wrapper::CudaContext_t context) {
    context_ = context;
  }

  cuda_wrapper::CudaContext_t lastContext() const noexcept {
    return last_context_;
  }

  cuda_wrapper::CudaModule_t loadModuleFromFile(const char *) override {
    return nullptr;
  }

  cuda_wrapper::CudaModule_t loadModuleFromImage(const void *) override {
    return nullptr;
  }

  cuda_wrapper::CudaFunction_t getFunction(cuda_wrapper::CudaModule_t,
                                           const char *name) override {
    if (g_hooks) {
      ++g_hooks->get_function_calls;
      g_hooks->last_kernel_name = name ? name : "";
    }
    return g_hooks ? g_hooks->function : nullptr;
  }

  void unloadModule(cuda_wrapper::CudaModule_t) override {}

private:
  cuda_wrapper::CudaContext_t context_{nullptr};
  cuda_wrapper::CudaContext_t last_context_{nullptr};
};

} // namespace

class CudaFunctionManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    g_hooks = &hooks_;
    slow_ops_ = std::make_unique<TestCudaSlowOps>();
    manager_ = std::make_unique<cuda_rt::CudaFunctionManager>();
    context_ = reinterpret_cast<cuda_wrapper::CudaContext_t>(0x1);
    module_ = reinterpret_cast<cuda_wrapper::CudaModule_t>(0x2);
    slow_ops_->setContextForTest(context_);
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
    slow_ops_.reset();
    g_hooks = nullptr;
  }

  void configureManager() {
    cuda_rt::CudaFunctionManager::Config config{};
    manager_->configureForTest(config, context_, module_, slow_ops_.get());
  }

  TestFunctionHooks hooks_{};
  std::unique_ptr<TestCudaSlowOps> slow_ops_;
  std::unique_ptr<cuda_rt::CudaFunctionManager> manager_;
  cuda_wrapper::CudaContext_t context_{nullptr};
  cuda_wrapper::CudaModule_t module_{nullptr};
};

TEST_F(CudaFunctionManagerTest, ConfigureSucceeds) {
  configureManager();
  EXPECT_TRUE(manager_->isConfiguredForTest());
}

TEST_F(CudaFunctionManagerTest, GetFunctionCachesByName) {
  configureManager();

  auto fn1 = manager_->getFunction("kernel_a");
  auto fn2 = manager_->getFunction("kernel_a");

  EXPECT_EQ(fn1, fn2);
  EXPECT_EQ(hooks_.get_function_calls, 1);
  EXPECT_EQ(hooks_.last_kernel_name, "kernel_a");
  EXPECT_EQ(slow_ops_->lastContext(), context_);
}

TEST_F(CudaFunctionManagerTest, InvalidNameThrows) {
  configureManager();
  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      [this] { manager_->getFunction(""); });
}

TEST_F(CudaFunctionManagerTest, NotConfiguredThrows) {
  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      [this] { manager_->getFunction("kernel_a"); });
}

TEST_F(CudaFunctionManagerTest, InvalidHandleThrows) {
  configureManager();
  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      [this] {
        manager_->acquire(cuda_rt::CudaFunctionManager::FunctionHandle::invalid());
      });
}

#endif // ORTEAF_ENABLE_CUDA
