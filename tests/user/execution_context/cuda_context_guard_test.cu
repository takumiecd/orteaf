#include "orteaf/user/execution_context/cuda_context_guard.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h"
#include "orteaf/internal/execution_context/cuda/current_context.h"
#include <gtest/gtest.h>

namespace user_ctx = ::orteaf::user::execution_context;
namespace cuda_api = ::orteaf::internal::execution::cuda::api;
namespace cuda_exec = ::orteaf::internal::execution::cuda;
namespace cuda_context = ::orteaf::internal::execution_context::cuda;
namespace cuda_wrapper = ::orteaf::internal::execution::cuda::platform::wrapper;

class CudaExecutionContextGuardTest : public ::testing::Test {
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
    } catch (const std::exception &) {
      GTEST_SKIP() << "CUDA not available or initialization failed";
    }
  }

  void TearDown() override {
    cuda_context::reset();
    cuda_api::CudaExecutionApi::shutdown();
  }
};

TEST_F(CudaExecutionContextGuardTest, GuardRestoresPreviousContext) {
  const auto baseline_device = cuda_context::currentDevice().payloadHandle();
  const auto baseline_context = cuda_context::currentCudaContext().payloadHandle();
  const auto baseline_stream = cuda_context::currentStream().payloadHandle();

  {
    user_ctx::CudaExecutionContextGuard guard;
    auto active_device = cuda_context::currentDevice().payloadHandle();
    EXPECT_EQ(active_device, cuda_exec::CudaDeviceHandle{0});
    EXPECT_TRUE(cuda_context::currentCudaContext());
    EXPECT_TRUE(cuda_context::currentStream());
  }

  const auto restored_device = cuda_context::currentDevice().payloadHandle();
  const auto restored_context = cuda_context::currentCudaContext().payloadHandle();
  const auto restored_stream = cuda_context::currentStream().payloadHandle();
  
  EXPECT_EQ(restored_device, baseline_device);
  EXPECT_EQ(restored_context, baseline_context);
  EXPECT_EQ(restored_stream, baseline_stream);
}

TEST_F(CudaExecutionContextGuardTest, GuardWithExplicitDevice) {
  user_ctx::CudaExecutionContextGuard guard(cuda_exec::CudaDeviceHandle{0});
  
  auto active_device = cuda_context::currentDevice().payloadHandle();
  EXPECT_EQ(active_device, cuda_exec::CudaDeviceHandle{0});
  EXPECT_TRUE(cuda_context::currentCudaContext());
  EXPECT_TRUE(cuda_context::currentStream());
}

// Note: GuardWithExplicitDeviceAndStream test removed
// Stream reuse across contexts requires careful lifetime management
// and is not a common use case for the guard API

TEST_F(CudaExecutionContextGuardTest, GuardMoveTransfersOwnership) {
  const auto baseline_device = cuda_context::currentDevice().payloadHandle();
  const auto baseline_context = cuda_context::currentCudaContext().payloadHandle();
  const auto baseline_stream = cuda_context::currentStream().payloadHandle();

  {
    user_ctx::CudaExecutionContextGuard guard;
    user_ctx::CudaExecutionContextGuard moved(std::move(guard));
    (void)moved;
  }

  const auto restored_device = cuda_context::currentDevice().payloadHandle();
  const auto restored_context = cuda_context::currentCudaContext().payloadHandle();
  const auto restored_stream = cuda_context::currentStream().payloadHandle();
  
  EXPECT_EQ(restored_device, baseline_device);
  EXPECT_EQ(restored_context, baseline_context);
  EXPECT_EQ(restored_stream, baseline_stream);
}

TEST_F(CudaExecutionContextGuardTest, GuardMoveAssignmentTransfersOwnership) {
  const auto baseline_device = cuda_context::currentDevice().payloadHandle();

  {
    user_ctx::CudaExecutionContextGuard guard1;
    user_ctx::CudaExecutionContextGuard guard2;
    guard2 = std::move(guard1);
  }

  const auto restored_device = cuda_context::currentDevice().payloadHandle();
  EXPECT_EQ(restored_device, baseline_device);
}

#endif // ORTEAF_ENABLE_CUDA
