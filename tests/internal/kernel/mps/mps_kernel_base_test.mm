#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>

namespace mps_kernel = orteaf::internal::kernel::mps;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;
namespace mps_context = orteaf::internal::execution_context::mps;

namespace {

TEST(MpsKernelBaseTest, DefaultConstruction) {
  mps_kernel::MpsKernelBase base;
  EXPECT_EQ(base.kernelCount(), 0u);
}

TEST(MpsKernelBaseTest, AddMultipleKeys) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");
  base.addKey("lib2", "func2");
  EXPECT_EQ(base.kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, AddKey) {
  mps_kernel::MpsKernelBase base;
  EXPECT_EQ(base.kernelCount(), 0u);

  base.addKey("library1", "function1");
  EXPECT_EQ(base.kernelCount(), 1u);

  base.addKey("library2", "function2");
  EXPECT_EQ(base.kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ReserveKeys) {
  mps_kernel::MpsKernelBase base;
  base.reserveKeys(10);
  EXPECT_EQ(base.kernelCount(), 0u);

  for (std::size_t i = 0; i < 10; ++i) {
    base.addKey("lib", "func");
  }
  EXPECT_EQ(base.kernelCount(), 10u);
}

TEST(MpsKernelBaseTest, ConfiguredReturnsFalseForUnconfiguredDevice) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");
  auto device = mps_exec::MpsDeviceHandle{42};
  EXPECT_FALSE(base.configured(device));
}

TEST(MpsKernelBaseTest, ConfigureWithMultipleKernels) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");
  base.addKey("lib2", "func2");

  EXPECT_EQ(base.kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ConfigureMultipleDevices) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");

  EXPECT_EQ(base.kernelCount(), 1u);
}

TEST(MpsKernelBaseTest, GetPipelineReturnsNullptrForUnconfiguredDevice) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");
  auto device = mps_exec::MpsDeviceHandle{42};

  EXPECT_EQ(base.getPipeline(device, 0), nullptr);
}

#if ORTEAF_ENABLE_TESTING
TEST(MpsKernelBaseTest, TestingHelpers) {
  mps_kernel::MpsKernelBase base;
  base.addKey("lib1", "func1");
  base.addKey("lib2", "func2");

  EXPECT_EQ(base.deviceCountForTest(), 0u);

  auto &keys = base.keysForTest();
  EXPECT_EQ(keys.size(), 2u);
}
#endif

} // namespace
