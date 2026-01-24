#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>

namespace mps_kernel = orteaf::internal::kernel::mps;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;

namespace {

// Mock RuntimeApi for testing initialize()
struct MockRuntimeApi {
  using PipelineLease = mps_kernel::MpsKernelBase::PipelineLease;
  using LibraryKey = mps_kernel::MpsKernelBase::LibraryKey;
  using FunctionKey = mps_kernel::MpsKernelBase::FunctionKey;

  static PipelineLease acquirePipeline(mps_exec::MpsDeviceHandle device,
                                       const LibraryKey &lib,
                                       const FunctionKey &func) {
    // Return empty lease for testing
    return PipelineLease{};
  }
};

TEST(MpsKernelBaseTest, DefaultConstruction) {
  mps_kernel::MpsKernelBase base;
  EXPECT_EQ(base.kernelCount(), 0u);
}

TEST(MpsKernelBaseTest, InitializerListConstruction) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}, {"lib2", "func2"}});
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

TEST(MpsKernelBaseTest, InitializedReturnsFalseForUninitializedDevice) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};
  EXPECT_FALSE(base.initialized(device));
}

TEST(MpsKernelBaseTest, InitializeMarksDeviceAsInitialized) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  EXPECT_FALSE(base.initialized(device));
  base.initialize<MockRuntimeApi>(device);
  EXPECT_TRUE(base.initialized(device));
}

TEST(MpsKernelBaseTest, InitializeWithMultipleKernels) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}, {"lib2", "func2"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);
  EXPECT_TRUE(base.initialized(device));
  EXPECT_EQ(base.kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, InitializeMultipleDevices) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device1 = mps_exec::MpsDeviceHandle{42};
  auto device2 = mps_exec::MpsDeviceHandle{43};

  base.initialize<MockRuntimeApi>(device1);
  base.initialize<MockRuntimeApi>(device2);

  EXPECT_TRUE(base.initialized(device1));
  EXPECT_TRUE(base.initialized(device2));
}

TEST(MpsKernelBaseTest, GetPipelineReturnsNullptrForUninitializedDevice) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  EXPECT_EQ(base.getPipeline(device, 0), nullptr);
}

TEST(MpsKernelBaseTest, GetPipelineReturnsNullptrForInvalidIndex) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);
  EXPECT_EQ(base.getPipeline(device, 999), nullptr);
}

TEST(MpsKernelBaseTest, GetPipelineReturnsValidPointerAfterInitialize) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);
  auto *pipeline = base.getPipeline(device, 0);
  EXPECT_NE(pipeline, nullptr);
}

TEST(MpsKernelBaseTest, GetPipelineConstVersion) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);
  const auto &const_base = base;
  const auto *pipeline = const_base.getPipeline(device, 0);
  EXPECT_NE(pipeline, nullptr);
}

TEST(MpsKernelBaseTest, GetPipelineForMultipleKernels) {
  mps_kernel::MpsKernelBase base(
      {{"lib1", "func1"}, {"lib2", "func2"}, {"lib3", "func3"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);

  auto *pipeline0 = base.getPipeline(device, 0);
  auto *pipeline1 = base.getPipeline(device, 1);
  auto *pipeline2 = base.getPipeline(device, 2);

  EXPECT_NE(pipeline0, nullptr);
  EXPECT_NE(pipeline1, nullptr);
  EXPECT_NE(pipeline2, nullptr);
}

TEST(MpsKernelBaseTest, ReinitializeClearsAndRecreatesPipelines) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}});
  auto device = mps_exec::MpsDeviceHandle{42};

  base.initialize<MockRuntimeApi>(device);
  EXPECT_TRUE(base.initialized(device));

  // Re-initialize should work
  base.initialize<MockRuntimeApi>(device);
  EXPECT_TRUE(base.initialized(device));
}

#if ORTEAF_ENABLE_TESTING
TEST(MpsKernelBaseTest, TestingHelpers) {
  mps_kernel::MpsKernelBase base({{"lib1", "func1"}, {"lib2", "func2"}});
  auto device1 = mps_exec::MpsDeviceHandle{42};
  auto device2 = mps_exec::MpsDeviceHandle{43};

  EXPECT_EQ(base.deviceCountForTest(), 0u);

  base.initialize<MockRuntimeApi>(device1);
  EXPECT_EQ(base.deviceCountForTest(), 1u);

  base.initialize<MockRuntimeApi>(device2);
  EXPECT_EQ(base.deviceCountForTest(), 2u);

  auto &keys = base.keysForTest();
  EXPECT_EQ(keys.size(), 2u);
}
#endif

} // namespace
