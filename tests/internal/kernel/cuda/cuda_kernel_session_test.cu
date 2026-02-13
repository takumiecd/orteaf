#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_types.h"
#include "orteaf/internal/execution_context/cuda/current_context.h"
#include "orteaf/internal/kernel/core/context_any.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/cuda/cuda_kernel_session.h"

#if ORTEAF_ENABLE_CUDA

namespace kernel = ::orteaf::internal::kernel;
namespace cuda_api = ::orteaf::internal::execution::cuda::api;
namespace cuda_context = ::orteaf::internal::execution_context::cuda;
namespace cuda_kernel = ::orteaf::internal::kernel::cuda;
namespace cuda_platform = ::orteaf::internal::execution::cuda::platform;
namespace cuda_wrapper = ::orteaf::internal::execution::cuda::platform::wrapper;

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

  cuda_wrapper::CudaStream_t createStream() override { return stream_; }
  void destroyStream(cuda_wrapper::CudaStream_t) override {}

  cuda_wrapper::CudaEvent_t createEvent() override { return event_; }
  void destroyEvent(cuda_wrapper::CudaEvent_t) override {}

  cuda_wrapper::CudaModule_t loadModuleFromFile(const char *) override {
    return module_;
  }

  cuda_wrapper::CudaModule_t loadModuleFromImage(const void *) override {
    return module_;
  }

  cuda_wrapper::CudaFunction_t getFunction(cuda_wrapper::CudaModule_t,
                                           const char *) override {
    return function_;
  }

  void unloadModule(cuda_wrapper::CudaModule_t) override {}

private:
  cuda_wrapper::CudaContext_t context_{
      reinterpret_cast<cuda_wrapper::CudaContext_t>(0x1)};
  cuda_wrapper::CudaStream_t stream_{
      reinterpret_cast<cuda_wrapper::CudaStream_t>(0x2)};
  cuda_wrapper::CudaEvent_t event_{
      reinterpret_cast<cuda_wrapper::CudaEvent_t>(0x3)};
  cuda_wrapper::CudaModule_t module_{
      reinterpret_cast<cuda_wrapper::CudaModule_t>(0x4)};
  cuda_wrapper::CudaFunction_t function_{
      reinterpret_cast<cuda_wrapper::CudaFunction_t>(0x5)};
};

cuda_api::CudaExecutionApi::ExecutionManager::Config makeCudaConfig(
    cuda_platform::CudaSlowOps *ops) {
  cuda_api::CudaExecutionApi::ExecutionManager::Config config{};
  config.slow_ops = ops;
  config.device_config.control_block_capacity = 1;
  config.device_config.control_block_block_size = 1;
  config.device_config.payload_capacity = 1;
  config.device_config.payload_block_size = 1;

  auto &context_cfg = config.device_config.context_config;
  context_cfg.control_block_capacity = 1;
  context_cfg.control_block_block_size = 1;
  context_cfg.payload_capacity = 1;
  context_cfg.payload_block_size = 1;

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

class CudaKernelSessionTest : public ::testing::Test {
protected:
  void SetUp() override {
    cuda_api::CudaExecutionApi::shutdown();
    cuda_api::CudaExecutionApi::configure(makeCudaConfig(new TestCudaSlowOps()));
  }

  void TearDown() override {
    cuda_context::reset();
    cuda_api::CudaExecutionApi::shutdown();
  }
};

TEST(CudaKernelSessionBasicTest, BeginReturnsNulloptWithInvalidContext) {
  cuda_api::CudaExecutionApi::shutdown();

  kernel::KernelArgs args;
  ::orteaf::internal::execution::cuda::resource::CudaKernelBase base;

  auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST(CudaKernelSessionBasicTest, BeginReturnsNulloptWithEmptyCudaContext) {
  cuda_api::CudaExecutionApi::shutdown();

  auto ctx_any = kernel::ContextAny::erase(cuda_context::Context{});
  kernel::KernelArgs args(std::move(ctx_any));
  ::orteaf::internal::execution::cuda::resource::CudaKernelBase base;

  auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST_F(CudaKernelSessionTest, BeginReturnsNulloptWhenFunctionIsUnavailable) {
  kernel::KernelArgs args(
      kernel::ContextAny::erase(cuda_context::currentContext()));
  ::orteaf::internal::execution::cuda::resource::CudaKernelBase base;

  auto session = cuda_kernel::CudaKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST_F(CudaKernelSessionTest, BeginSucceedsWithConfiguredFunctionLease) {
  kernel::KernelArgs args(
      kernel::ContextAny::erase(cuda_context::currentContext()));
  auto *context = args.context().tryAs<cuda_context::Context>();
  ASSERT_NE(context, nullptr);

  cuda_api::CudaExecutionApi::KernelKeys keys;
  keys.pushBack(
      {cuda_api::CudaExecutionApi::ModuleKey::File("module.bin"), "kernel_a"});

  auto lease =
      cuda_api::CudaExecutionApi::executionManager().kernelBaseManager().acquire(
          keys);
  ASSERT_TRUE(lease);
  auto *base = lease.operator->();
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(base->ensureFunctions(context->context));

  auto session = cuda_kernel::CudaKernelSession::begin(*base, args, 0);
  ASSERT_TRUE(session.has_value());
  EXPECT_TRUE(session->functionLease());
  EXPECT_NE(session->function(), nullptr);
  EXPECT_NE(session->stream(), nullptr);
}

TEST(CudaKernelSessionBasicTest, MakeGridAndBlock1DHelpers) {
  const auto block = cuda_kernel::CudaKernelSession::makeBlock1D(128);
  EXPECT_EQ(block.x, 128u);
  EXPECT_EQ(block.y, 1u);
  EXPECT_EQ(block.z, 1u);

  const auto grid = cuda_kernel::CudaKernelSession::makeGrid1D(1000, 256);
  EXPECT_EQ(grid.x, 4u);
  EXPECT_EQ(grid.y, 1u);
  EXPECT_EQ(grid.z, 1u);
}

} // namespace

#endif // ORTEAF_ENABLE_CUDA
