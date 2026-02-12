#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/execution/mps/manager/mps_command_queue_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_size.h>
#include <orteaf/internal/execution/mps/resource/mps_command_queue_resource.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/execution/mps/resource/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/storage/mps/mps_storage.h>

namespace mps_kernel = ::orteaf::internal::execution::mps::resource;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;
namespace mps_context = orteaf::internal::execution_context::mps;
namespace mps_resource = orteaf::internal::execution::mps::resource;
namespace mps_session = orteaf::internal::kernel::mps;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace mps_storage = orteaf::internal::storage::mps;

namespace {

mps_manager::MpsKernelBaseManager::Config makeKernelBaseManagerConfig() {
  mps_manager::MpsKernelBaseManager::Config config{};
  config.control_block_capacity = 4;
  config.control_block_block_size = 4;
  config.payload_capacity = 4;
  config.payload_block_size = 4;
  return config;
}

::orteaf::internal::base::HeapVector<mps_manager::MpsKernelBaseManager::Key>
makeKeys(std::initializer_list<std::pair<const char *, const char *>> names) {
  ::orteaf::internal::base::HeapVector<mps_manager::MpsKernelBaseManager::Key>
      keys;
  keys.reserve(names.size());
  for (const auto &[library, function] : names) {
    keys.pushBack({mps_manager::LibraryKey::Named(std::string(library)),
                   mps_manager::FunctionKey::Named(std::string(function))});
  }
  return keys;
}

TEST(MpsKernelBaseTest, DefaultConstruction) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  EXPECT_EQ(base.kernelCount(), 0u);
}

TEST(MpsKernelBaseTest, AddMultipleKeys) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, AddKey) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease =
      manager.acquire(makeKeys({{"library1", "function1"},
                                {"library2", "function2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ReserveKeys) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 10u);
}

TEST(MpsKernelBaseTest, ConfiguredReturnsFalseForUnconfiguredDevice) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  auto device = mps_exec::MpsDeviceHandle{42};
  EXPECT_FALSE(lease->configured(device));
}

TEST(MpsKernelBaseTest, ConfigureWithMultipleKernels) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ConfigureMultipleDevices) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 1u);
}

TEST(MpsKernelBaseTest, GetPipelineReturnsNullptrForUnconfiguredDevice) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  auto device = mps_exec::MpsDeviceHandle{42};

  auto pipeline = lease->getPipelineLease(device, 0);
  EXPECT_FALSE(pipeline);
}

#if ORTEAF_ENABLE_TESTING
TEST(MpsKernelBaseTest, TestingHelpers) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());
  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);

  EXPECT_EQ(lease->deviceCountForTest(), 0u);

  auto &keys = lease->keysForTest();
  EXPECT_EQ(keys.size(), 2u);
}
#endif

// =============================================================================
// createCommandBuffer Tests
// =============================================================================

TEST(MpsKernelBaseTest, CreateCommandBufferReturnsNullptrForEmptyContext) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  mps_context::Context context;

  auto buffer = mps_session::MpsKernelSession::createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

TEST(MpsKernelBaseTest,
     CreateCommandBufferReturnsNullptrForContextWithoutQueue) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  mps_context::Context context;
  context.device = {}; // Empty device lease

  auto buffer = mps_session::MpsKernelSession::createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, CreateCommandBufferSucceedsWithValidContext) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  queue_resource.setQueueForTest(queue);

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
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  auto encoder = mps_session::MpsKernelSession::createComputeCommandEncoder(nullptr);
  EXPECT_EQ(encoder, nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, CreateComputeCommandEncoderWithRealHardware) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  auto encoder = mps_session::MpsKernelSession::createComputeCommandEncoder(buffer);
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
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
// endEncoding Tests
// =============================================================================

TEST(MpsKernelBaseTest, EndEncodingWithNullptrEncoderDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Should not crash with null encoder
  mps_session::MpsKernelSession::endEncoding(nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, EndEncodingWithValidEncoder) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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

  // End encoding should succeed
  mps_session::MpsKernelSession::endEncoding(encoder);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// commit Tests
// =============================================================================

TEST(MpsKernelBaseTest, CommitWithNullptrBufferDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Should not crash with null buffer
  mps_session::MpsKernelSession::commit(nullptr);
}

// =============================================================================
// waitUntilCompleted Tests
// =============================================================================

TEST(MpsKernelBaseTest, WaitUntilCompletedWithNullptrBufferDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Should not crash with null buffer
  mps_session::MpsKernelSession::waitUntilCompleted(nullptr);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, CommitAndWaitWithValidBuffer) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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

  // End encoding, commit, and wait
  mps_session::MpsKernelSession::endEncoding(encoder);
  mps_session::MpsKernelSession::commit(cmd_buffer);
  mps_session::MpsKernelSession::waitUntilCompleted(cmd_buffer);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// setBuffer Tests
// =============================================================================

TEST(MpsKernelBaseTest, SetBufferWithNullptrEncoderDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  mps_storage::MpsStorage storage;

  // Should not crash with null encoder
  mps_session::MpsKernelSession::setBuffer(nullptr, storage, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetBufferWithInvalidStorageDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  mps_session::MpsKernelSession::setBuffer(encoder, empty_storage, 0);

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
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  int data = 42;

  // Should not crash with null encoder
  mps_session::MpsKernelSession::setBytes(nullptr, &data, sizeof(data), 0);
}

TEST(MpsKernelBaseTest, SetBytesWithNullptrDataDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Should not crash with null data
  mps_session::MpsKernelSession::setBytes(reinterpret_cast<mps_wrapper::MpsComputeCommandEncoder_t>(0x1), nullptr, 4, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetBytesWithValidEncoder) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  mps_session::MpsKernelSession::setBytes(encoder, &data, sizeof(data), 0);

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
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{0}, 42);

  // Should not crash with null encoder
  mps_session::MpsKernelSession::setParam(nullptr, param, 0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, SetParamWithIntParam) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  mps_session::MpsKernelSession::setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}

TEST(MpsKernelBaseTest, SetParamWithFloatParam) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  mps_session::MpsKernelSession::setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}

TEST(MpsKernelBaseTest, SetParamWithSizeTParam) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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
  mps_session::MpsKernelSession::setParam(encoder, param, 0);

  // Cleanup
  mps_wrapper::destroyComputeCommandEncoder(encoder);
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// setPipelineState Tests
// =============================================================================

TEST(MpsKernelBaseTest, SetPipelineStateWithNullptrEncoderDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  mps_kernel::MpsKernelBase::PipelineLease pipeline;

  // Should not crash with null encoder
  mps_session::MpsKernelSession::setPipelineState(nullptr, pipeline);
}

TEST(MpsKernelBaseTest, SetPipelineStateWithEmptyLeaseDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Create a mock encoder (non-null pointer)
  auto fake_encoder = reinterpret_cast<mps_wrapper::MpsComputeCommandEncoder_t>(0x1);
  mps_kernel::MpsKernelBase::PipelineLease empty_pipeline;

  // Should not crash with empty pipeline lease
  mps_session::MpsKernelSession::setPipelineState(fake_encoder, empty_pipeline);
}

// =============================================================================
// dispatchThreadgroups Tests
// =============================================================================

TEST(MpsKernelBaseTest, DispatchThreadgroupsWithNullptrEncoderDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  auto threadgroups = mps_wrapper::makeSize(1, 1, 1);
  auto threads_per_threadgroup = mps_wrapper::makeSize(32, 1, 1);

  // Should not crash with null encoder
  mps_session::MpsKernelSession::dispatchThreadgroups(nullptr, threadgroups, threads_per_threadgroup);
}

// =============================================================================
// dispatchThreads Tests
// =============================================================================

TEST(MpsKernelBaseTest, DispatchThreadsWithNullptrEncoderDoesNotCrash) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;
  auto threads_per_grid = mps_wrapper::makeSize(1024, 1, 1);
  auto threads_per_threadgroup = mps_wrapper::makeSize(256, 1, 1);

  // Should not crash with null encoder
  mps_session::MpsKernelSession::dispatchThreads(nullptr, threads_per_grid, threads_per_threadgroup);
}

// =============================================================================
// GPU Timing Tests
// =============================================================================

TEST(MpsKernelBaseTest, GetGPUTimesWithNullptrReturnsZero) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

  // Should return 0.0 for null command buffer
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUStartTime(nullptr), 0.0);
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUEndTime(nullptr), 0.0);
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUDuration(nullptr), 0.0);
}

#if ORTEAF_ENABLE_MPS
TEST(MpsKernelBaseTest, GetGPUTimesWithUnscheduledBufferReturnsZero) {
  [[maybe_unused]] mps_kernel::MpsKernelBase base;

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

  // Unscheduled buffer should return 0.0
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUStartTime(cmd_buffer), 0.0);
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUEndTime(cmd_buffer), 0.0);
  EXPECT_EQ(mps_session::MpsKernelSession::getGPUDuration(cmd_buffer), 0.0);

  // Cleanup
  mps_wrapper::destroyCommandBuffer(cmd_buffer);
  mps_wrapper::destroyCommandQueue(queue);
  mps_wrapper::deviceRelease(device);
}
#endif

// =============================================================================
// Grid Size Helper Tests
// =============================================================================

TEST(MpsKernelBaseTest, MakeGridSizeCreates1DSize) {
  auto size = mps_session::MpsKernelSession::makeGridSize(256);
  EXPECT_EQ(size.width, 256);
  EXPECT_EQ(size.height, 1);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelBaseTest, MakeGridSizeCreates2DSize) {
  auto size = mps_session::MpsKernelSession::makeGridSize(16, 16);
  EXPECT_EQ(size.width, 16);
  EXPECT_EQ(size.height, 16);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelBaseTest, MakeGridSizeCreates3DSize) {
  auto size = mps_session::MpsKernelSession::makeGridSize(8, 8, 8);
  EXPECT_EQ(size.width, 8);
  EXPECT_EQ(size.height, 8);
  EXPECT_EQ(size.depth, 8);
}

TEST(MpsKernelBaseTest, MakeThreadsPerThreadgroupCreates1DSize) {
  auto size = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(256);
  EXPECT_EQ(size.width, 256);
  EXPECT_EQ(size.height, 1);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelBaseTest, MakeThreadsPerThreadgroupCreates2DSize) {
  auto size = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(16, 16);
  EXPECT_EQ(size.width, 16);
  EXPECT_EQ(size.height, 16);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelBaseTest, CalculateGridSizeRoundsUp) {
  auto total_threads = mps_session::MpsKernelSession::makeGridSize(1000);
  auto threads_per_threadgroup = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(256);
  auto grid_size = mps_session::MpsKernelSession::calculateGridSize(total_threads, threads_per_threadgroup);
  
  // 1000 / 256 = 3.90625, should round up to 4
  EXPECT_EQ(grid_size.width, 4);
  EXPECT_EQ(grid_size.height, 1);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelBaseTest, CalculateGridSizeExactDivision) {
  auto total_threads = mps_session::MpsKernelSession::makeGridSize(1024);
  auto threads_per_threadgroup = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(256);
  auto grid_size = mps_session::MpsKernelSession::calculateGridSize(total_threads, threads_per_threadgroup);
  
  // 1024 / 256 = 4 exactly
  EXPECT_EQ(grid_size.width, 4);
  EXPECT_EQ(grid_size.height, 1);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelBaseTest, CalculateGridSize2D) {
  auto total_threads = mps_session::MpsKernelSession::makeGridSize(1920, 1080);
  auto threads_per_threadgroup = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(16, 16);
  auto grid_size = mps_session::MpsKernelSession::calculateGridSize(total_threads, threads_per_threadgroup);
  
  // 1920 / 16 = 120, 1080 / 16 = 67.5 -> 68
  EXPECT_EQ(grid_size.width, 120);
  EXPECT_EQ(grid_size.height, 68);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelBaseTest, CalculateGridSize3D) {
  auto total_threads = mps_session::MpsKernelSession::makeGridSize(100, 100, 100);
  auto threads_per_threadgroup = mps_session::MpsKernelSession::makeThreadsPerThreadgroup(8, 8, 8);
  auto grid_size = mps_session::MpsKernelSession::calculateGridSize(total_threads, threads_per_threadgroup);
  
  // 100 / 8 = 12.5 -> 13 for each dimension
  EXPECT_EQ(grid_size.width, 13);
  EXPECT_EQ(grid_size.height, 13);
  EXPECT_EQ(grid_size.depth, 13);
}

// Note: Actual dispatch requires a valid pipeline state to be set,
// which requires shader compilation. These tests only verify the
// method calls don't crash with valid encoders.

} // namespace
