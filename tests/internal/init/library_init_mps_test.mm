#if ORTEAF_ENABLE_MPS

#include <gtest/gtest.h>

#include "orteaf/internal/init/library_init.h"
#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"

namespace init = ::orteaf::internal::init;
namespace mps_api = ::orteaf::internal::execution::mps::api;
namespace mps_exec = ::orteaf::internal::execution::mps;
namespace mps_platform = ::orteaf::internal::execution::mps::platform;

class LibraryInitMpsTest : public ::testing::Test {
protected:
  void TearDown() override { init::shutdown(); }
};

TEST_F(LibraryInitMpsTest, InitializeConfiguresMpsExecution) {
  mps_platform::MpsSlowOpsImpl ops;
  const int device_count = ops.getDeviceCount();
  if (device_count <= 0) {
    GTEST_SKIP() << "No MPS devices available";
  }

  EXPECT_NO_THROW(init::initialize());
  EXPECT_TRUE(init::isInitialized());

  auto device = mps_api::MpsExecutionApi::acquireDevice(
      mps_exec::MpsDeviceHandle{0});
  EXPECT_TRUE(device);
}

#endif  // ORTEAF_ENABLE_MPS
