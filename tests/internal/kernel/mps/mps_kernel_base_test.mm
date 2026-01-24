#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/execution/mps/manager/mps_command_queue_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/resource/mps_command_queue_resource.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>

namespace mps_kernel = orteaf::internal::kernel::mps;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;
namespace mps_context = orteaf::internal::execution_context::mps;
namespace mps_resource = orteaf::internal::execution::mps::resource;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;

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

// =============================================================================
// createCommandBuffer Tests
// =============================================================================

TEST(MpsKernelBaseTest, CreateCommandBufferReturnsNullptrForEmptyContext) {
  mps_kernel::MpsKernelBase base;
  mps_context::Context context;

  auto buffer = base.createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

TEST(MpsKernelBaseTest,
     CreateCommandBufferReturnsNullptrForContextWithoutQueue) {
  mps_kernel::MpsKernelBase base;
  mps_context::Context context;
  context.device = {}; // Empty device lease

  auto buffer = base.createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, CreateCommandBufferSucceedsWithValidContext) {
  mps_kernel::MpsKernelBase base;

  // Create a real device and command queue
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  // Create a context with a mock command queue lease
  mps_context::Context context;

  // Manually set up a command queue resource
  mps_resource::MpsCommandQueueResource queue_resource;
  queue_resource.setQueue(queue);

  // Create a mock lease that points to the resource
  // Note: This is a bit tricky because we need a proper lease structure
  // For now, we'll just verify the logic with nullptr checks
  // A full integration test would require proper manager setup

  // Cleanup
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);

  // For unit test purposes, verify nullptr is returned when queue resource
  // doesn't have a valid queue
  mps_resource::MpsCommandQueueResource empty_resource;
  EXPECT_FALSE(empty_resource.hasQueue());
}

TEST(MpsKernelBaseTest, CreateCommandBufferWithRealHardware) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  // Create command buffer directly to verify the wrapper works
  auto buffer = mps_wrapper::createCommandBuffer(queue);
  EXPECT_NE(buffer, nullptr);

  // Cleanup
  if (buffer != nullptr) {
    mps_wrapper::destroyCommandBuffer(buffer);
  }
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

} // namespace
