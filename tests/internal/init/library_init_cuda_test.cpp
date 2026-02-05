#if ORTEAF_ENABLE_CUDA

#include <gtest/gtest.h>

#include <exception>

#include "orteaf/internal/init/library_init.h"
#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h"
#include "orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h"

namespace init = ::orteaf::internal::init;
namespace cuda_api = ::orteaf::internal::execution::cuda::api;
namespace cuda_exec = ::orteaf::internal::execution::cuda;
namespace cuda_wrapper =
    ::orteaf::internal::execution::cuda::platform::wrapper;

class LibraryInitCudaTest : public ::testing::Test {
protected:
  void TearDown() override { init::shutdown(); }
};

TEST_F(LibraryInitCudaTest, InitializeConfiguresCudaExecution) {
  int device_count = 0;
  try {
    cuda_wrapper::cudaInit();
    device_count = cuda_wrapper::getDeviceCount();
  } catch (const std::exception &ex) {
    GTEST_SKIP() << "CUDA init failed: " << ex.what();
  }

  EXPECT_NO_THROW(init::initialize());
  EXPECT_TRUE(init::isInitialized());

  if (device_count <= 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }

  auto device = cuda_api::CudaExecutionApi::acquireDevice(
      cuda_exec::CudaDeviceHandle{0});
  EXPECT_TRUE(device);
}

#endif  // ORTEAF_ENABLE_CUDA
