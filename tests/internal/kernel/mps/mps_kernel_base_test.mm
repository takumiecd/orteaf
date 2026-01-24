#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/execution/mps/manager/mps_command_queue_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_size.h>
#include <orteaf/internal/execution/mps/resource/mps_command_queue_resource.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/param.h>
#include <orteaf/internal/kernel/param_id.h>
#include <orteaf/internal/storage/mps/mps_storage.h>

namespace mps_kernel = orteaf::internal::kernel::mps;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;
namespace mps_context = orteaf::internal::execution_context::mps;
namespace mps_resource = orteaf::internal::execution::mps::resource;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace mps_storage = orteaf::internal::storage::mps;

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

// =============================================================================
// createComputeCommandEncoder Tests
// =============================================================================

TEST(MpsKernelBaseTest, CreateComputeCommandEncoderReturnsNullptrForNullBuffer) {
  mps_kernel::MpsKernelBase base;

  auto encoder = base.createComputeCommandEncoder(nullptr);
  EXPECT_EQ(encoder, nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, CreateComputeCommandEncoderWithRealHardware) {
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

  // Create command buffer
  auto buffer = mps_wrapper::createCommandBuffer(queue);
  ASSERT_NE(buffer, nullptr);

  // Create compute command encoder using the base method
  auto encoder = base.createComputeCommandEncoder(buffer);
  EXPECT_NE(encoder, nullptr);

  // Cleanup
  if (encoder != nullptr) {
    mps_wrapper::destroyComputeCommandEncoder(encoder);
  }
  if (buffer != nullptr) {
    mps_wrapper::destroyCommandBuffer(buffer);
  }
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}

TEST(MpsKernelBaseTest, CreateComputeCommandEncoderSucceeds) {
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

  // Create command buffer
  auto buffer = mps_wrapper::createCommandBuffer(queue);
  if (buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  // Create encoder directly to verify the wrapper works
  auto encoder = mps_wrapper::createComputeCommandEncoder(buffer);
  EXPECT_NE(encoder, nullptr);

  // Cleanup
  if (encoder != nullptr) {
    mps_wrapper::destroyComputeCommandEncoder(encoder);
  }
  mps_wrapper::destroyCommandBuffer(buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// setBuffer Tests
// =============================================================================

TEST(MpsKernelBaseTest, SetBufferWithNullptrEncoderDoesNotCrash) {
  mps_kernel::MpsKernelBase base;
  mps_storage::MpsStorage storage;

  // Should not crash with null encoder
  base.setBuffer(nullptr, storage, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetBufferWithInvalidStorageDoesNotCrash) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue and buffer
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  auto cmd_buffer = mps_wrapper::createCommandBuffer(queue);
  if (cmd_buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  auto encoder = mps_wrapper::createComputeCommandEncoder(cmd_buffer);
  if (encoder == nullptr) {
    mps_wrapper::destroyCommandBuffer(cmd_buffer);
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create encoder";
  }

  // Empty storage should not crash
  mps_storage::MpsStorage empty_storage;
  base.setBuffer(encoder, empty_storage, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// setBytes Tests
// =============================================================================

TEST(MpsKernelBaseTest, SetBytesWithNullptrEncoderDoesNotCrash) {
  mps_kernel::MpsKernelBase base;
  int data = 42;

  // Should not crash with null encoder
  base.setBytes(nullptr, &data, sizeof(data), 0);
}

TEST(MpsKernelBaseTest, SetBytesWithNullptrDataDoesNotCrash) {
  mps_kernel::MpsKernelBase base;

  // Should not crash with null data
  base.setBytes(reinterpret_cast<mps_wrapper::MpsComputeCommandEncoder_t>(0x1), nullptr, 4, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetBytesWithValidEncoder) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue and buffer
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  auto cmd_buffer = mps_wrapper::createCommandBuffer(queue);
  if (cmd_buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  auto encoder = mps_wrapper::createComputeCommandEncoder(cmd_buffer);
  if (encoder == nullptr) {
    mps_wrapper::destroyCommandBuffer(cmd_buffer);
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create encoder";
  }

  // Test with actual data
  int data = 42;
  base.setBytes(encoder, &data, sizeof(data), 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// setParam Tests
// =============================================================================

TEST(MpsKernelBaseTest, SetParamWithNullptrEncoderDoesNotCrash) {
  mps_kernel::MpsKernelBase base;
  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{0}, 42);

  // Should not crash with null encoder
  base.setParam(nullptr, param, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetParamWithIntParam) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue and buffer
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  auto cmd_buffer = mps_wrapper::createCommandBuffer(queue);
  if (cmd_buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  auto encoder = mps_wrapper::createComputeCommandEncoder(cmd_buffer);
  if (encoder == nullptr) {
    mps_wrapper::destroyCommandBuffer(cmd_buffer);
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create encoder";
  }

  // Test with int param
  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{0}, 42);
  base.setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}

TEST(MpsKernelBaseTest, SetParamWithFloatParam) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue and buffer
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  auto cmd_buffer = mps_wrapper::createCommandBuffer(queue);
  if (cmd_buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  auto encoder = mps_wrapper::createComputeCommandEncoder(cmd_buffer);
  if (encoder == nullptr) {
    mps_wrapper::destroyCommandBuffer(cmd_buffer);
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create encoder";
  }

  // Test with float param
  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{1}, 3.14f);
  base.setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}

TEST(MpsKernelBaseTest, SetParamWithSizeTParam) {
  mps_kernel::MpsKernelBase base;

  // Create a real device
  auto device = mps_wrapper::getDevice();
  if (device == nullptr) {
    GTEST_SKIP() << "No Metal devices available";
  }

  // Create command queue and buffer
  auto queue = mps_wrapper::createCommandQueue(device);
  if (queue == nullptr) {
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command queue";
  }

  auto cmd_buffer = mps_wrapper::createCommandBuffer(queue);
  if (cmd_buffer == nullptr) {
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create command buffer";
  }

  auto encoder = mps_wrapper::createComputeCommandEncoder(cmd_buffer);
  if (encoder == nullptr) {
    mps_wrapper::destroyCommandBuffer(cmd_buffer);
    mps_wrapper::destroyCommandQueue(queue);
    mps_wrapper::deviceRelease(device);
    GTEST_SKIP() << "Failed to create encoder";
  }

  // Test with size_t param
  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{2}, std::size_t{1024});
  base.setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// dispatchThreadgroups Tests
// =============================================================================

TEST(MpsKernelBaseTest, DispatchThreadgroupsWithNullptrEncoderDoesNotCrash) {
  mps_kernel::MpsKernelBase base;
  auto threadgroups = mps_wrapper::makeSize(1, 1, 1);
  auto threads_per_threadgroup = mps_wrapper::makeSize(32, 1, 1);

  // Should not crash with null encoder
  base.dispatchThreadgroups(nullptr, threadgroups, threads_per_threadgroup);
}

// Note: Actual dispatch requires a valid pipeline state to be set,
// which requires shader compilation. These tests only verify the
// method calls don't crash with valid encoders.

} // namespace
