#include "orteaf/internal/execution_context/cuda/current_context.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h"
#include <gtest/gtest.h>

namespace cuda_context = ::orteaf::internal::execution_context::cuda;
namespace cuda_api = ::orteaf::internal::execution::cuda::api;
namespace cuda_exec = ::orteaf::internal::execution::cuda;
namespace cuda_wrapper = ::orteaf::internal::execution::cuda::platform::wrapper;
namespace architecture = ::orteaf::internal::architecture;

class CudaCurrentContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Skip if no CUDA devices available
    try {
      cuda_wrapper::cudaInit();
      const int device_count = cuda_wrapper::getDeviceCount();
      if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
      }

      cuda_api::CudaExecutionApi::ExecutionManager::Config config{};
      const auto pool_size = static_cast<std::size_t>(device_count);
      const auto stream_pool_size = pool_size * 4;
      
      // Device manager config
      config.device_config.control_block_capacity = pool_size;
      config.device_config.control_block_block_size = pool_size;
      config.device_config.payload_capacity = pool_size;
      config.device_config.payload_block_size = pool_size;
      
      // Context manager config
      config.device_config.context_config.control_block_capacity = pool_size;
      config.device_config.context_config.control_block_block_size = pool_size;
      config.device_config.context_config.payload_capacity = pool_size;
      config.device_config.context_config.payload_block_size = pool_size;
      
      // Stream manager config
      config.device_config.context_config.stream_config.control_block_capacity = stream_pool_size;
      config.device_config.context_config.stream_config.control_block_block_size = stream_pool_size;
      config.device_config.context_config.stream_config.payload_capacity = stream_pool_size;
      config.device_config.context_config.stream_config.payload_block_size = stream_pool_size;
      
      // Event manager config
      config.device_config.context_config.event_config.control_block_capacity = stream_pool_size;
      config.device_config.context_config.event_config.control_block_block_size = stream_pool_size;
      config.device_config.context_config.event_config.payload_capacity = stream_pool_size;
      config.device_config.context_config.event_config.payload_block_size = stream_pool_size;
      
      // Buffer manager config
      config.device_config.context_config.buffer_config.control_block_capacity = stream_pool_size;
      config.device_config.context_config.buffer_config.control_block_block_size = stream_pool_size;
      config.device_config.context_config.buffer_config.payload_capacity = stream_pool_size;
      config.device_config.context_config.buffer_config.payload_block_size = stream_pool_size;
      
      // Module manager config
      config.device_config.context_config.module_config.control_block_capacity = pool_size;
      config.device_config.context_config.module_config.control_block_block_size = pool_size;
      config.device_config.context_config.module_config.payload_capacity = pool_size;
      config.device_config.context_config.module_config.payload_block_size = pool_size;
      
      cuda_api::CudaExecutionApi::configure(config);
      cuda_context::reset();
    } catch (const std::exception &e) {
      GTEST_SKIP() << "CUDA not available or initialization failed: " << e.what();
    }
  }

  void TearDown() override {
    cuda_context::reset();
    cuda_api::CudaExecutionApi::shutdown();
  }
};

TEST_F(CudaCurrentContextTest, CurrentContextProvidesDefaultResources) {
  const auto &ctx = cuda_context::currentContext();
  EXPECT_TRUE(ctx.device);
  EXPECT_TRUE(ctx.context);
  EXPECT_TRUE(ctx.stream);

  auto device = cuda_context::currentDevice();
  EXPECT_TRUE(device);
  EXPECT_EQ(ctx.device.payloadHandle(), device.payloadHandle());
  EXPECT_EQ(device.payloadHandle(), cuda_exec::CudaDeviceHandle{0});
}

TEST_F(CudaCurrentContextTest, CurrentCudaContextReturnsContext) {
  auto context = cuda_context::currentCudaContext();
  EXPECT_TRUE(context);
}

TEST_F(CudaCurrentContextTest, CurrentStreamReturnsStream) {
  auto stream = cuda_context::currentStream();
  EXPECT_TRUE(stream);
}

TEST_F(CudaCurrentContextTest, CurrentArchitectureMatchesCurrentContext) {
  const auto &ctx = cuda_context::currentContext();
  const auto arch = cuda_context::currentArchitecture();
  EXPECT_EQ(arch, ctx.architecture());
  EXPECT_EQ(architecture::executionOf(arch),
            ::orteaf::internal::execution::Execution::Cuda);
}

TEST_F(CudaCurrentContextTest, SetCurrentContextOverridesState) {
  cuda_context::Context ctx{cuda_exec::CudaDeviceHandle{0}};

  cuda_context::setCurrentContext(std::move(ctx));

  const auto &current_ctx = cuda_context::currentContext();
  EXPECT_TRUE(current_ctx.device);
  EXPECT_TRUE(current_ctx.context);
  EXPECT_TRUE(current_ctx.stream);
  EXPECT_EQ(current_ctx.device.payloadHandle(), cuda_exec::CudaDeviceHandle{0});
}

TEST_F(CudaCurrentContextTest, SetCurrentOverridesState) {
  cuda_context::Context ctx{cuda_exec::CudaDeviceHandle{0}};

  cuda_context::CurrentContext current{};
  current.current = std::move(ctx);
  cuda_context::setCurrent(std::move(current));

  const auto &current_ctx = cuda_context::currentContext();
  EXPECT_TRUE(current_ctx.device);
  EXPECT_TRUE(current_ctx.context);
  EXPECT_TRUE(current_ctx.stream);
  EXPECT_EQ(current_ctx.device.payloadHandle(), cuda_exec::CudaDeviceHandle{0});
}

TEST_F(CudaCurrentContextTest, ResetReacquiresDefaultResources) {
  auto first = cuda_context::currentDevice();
  EXPECT_TRUE(first);

  cuda_context::reset();

  auto second = cuda_context::currentDevice();
  EXPECT_TRUE(second);
  EXPECT_EQ(second.payloadHandle(), cuda_exec::CudaDeviceHandle{0});

  auto context = cuda_context::currentCudaContext();
  EXPECT_TRUE(context);

  auto stream = cuda_context::currentStream();
  EXPECT_TRUE(stream);
}

TEST_F(CudaCurrentContextTest, ContextConstructorAcquiresResources) {
  cuda_context::Context ctx{cuda_exec::CudaDeviceHandle{0}};
  
  EXPECT_TRUE(ctx.device);
  EXPECT_TRUE(ctx.context);
  EXPECT_TRUE(ctx.stream);
  EXPECT_EQ(ctx.device.payloadHandle(), cuda_exec::CudaDeviceHandle{0});
}

#endif // ORTEAF_ENABLE_CUDA
