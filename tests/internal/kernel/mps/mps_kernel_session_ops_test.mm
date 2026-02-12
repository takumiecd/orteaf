#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_size.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session_ops.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/storage/mps/mps_storage.h>

#include "tests/internal/kernel/mps/test_utils/mps_hardware_test_utils.h"

namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_ops = ::orteaf::internal::kernel::mps;
namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;
namespace mps_storage = ::orteaf::internal::storage::mps;
namespace mps_test_utils =
    ::orteaf::tests::internal::kernel::mps::test_utils;

namespace {

TEST(MpsKernelSessionOpsTest, CreateCommandBufferReturnsNullptrForEmptyContext) {
  mps_context::Context context;
  auto buffer = mps_ops::MpsKernelSessionOps::createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

TEST(MpsKernelSessionOpsTest,
     CreateCommandBufferReturnsNullptrForContextWithoutQueue) {
  mps_context::Context context;
  context.device = {};
  auto buffer = mps_ops::MpsKernelSessionOps::createCommandBuffer(context);
  EXPECT_EQ(buffer, nullptr);
}

TEST(MpsKernelSessionOpsTest,
     CreateComputeCommandEncoderReturnsNullptrForNullBuffer) {
  auto encoder =
      mps_ops::MpsKernelSessionOps::createComputeCommandEncoder(nullptr);
  EXPECT_EQ(encoder, nullptr);
}

TEST(MpsKernelSessionOpsTest, CreateComputeCommandEncoderWithRealHardware) {
  auto acquired = mps_test_utils::acquireHardware(true, false);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;
  auto encoder = mps_ops::MpsKernelSessionOps::createComputeCommandEncoder(
      hardware.command_buffer);
  EXPECT_NE(encoder, nullptr);
  if (encoder != nullptr) {
    mps_wrapper::destroyComputeCommandEncoder(encoder);
  }
}

TEST(MpsKernelSessionOpsTest, EndEncodingWithNullptrEncoderDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::endEncoding(nullptr);
}

TEST(MpsKernelSessionOpsTest, EndEncodingWithValidEncoder) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;
  mps_ops::MpsKernelSessionOps::endEncoding(hardware.encoder);
}

TEST(MpsKernelSessionOpsTest,
     DestroyComputeCommandEncoderWithNullptrDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::destroyComputeCommandEncoder(nullptr);
}

TEST(MpsKernelSessionOpsTest, CommitWithNullptrBufferDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::commit(nullptr);
}

TEST(MpsKernelSessionOpsTest, DestroyCommandBufferWithNullptrDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::destroyCommandBuffer(nullptr);
}

TEST(MpsKernelSessionOpsTest, WaitUntilCompletedWithNullptrBufferDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::waitUntilCompleted(nullptr);
}

TEST(MpsKernelSessionOpsTest, CommitAndWaitWithValidBuffer) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  mps_ops::MpsKernelSessionOps::endEncoding(hardware.encoder);
  mps_ops::MpsKernelSessionOps::commit(hardware.command_buffer);
  mps_ops::MpsKernelSessionOps::waitUntilCompleted(hardware.command_buffer);
}

TEST(MpsKernelSessionOpsTest, DestroyComputeCommandEncoderWithValidEncoder) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;
  mps_ops::MpsKernelSessionOps::destroyComputeCommandEncoder(hardware.encoder);
  hardware.encoder = nullptr;
}

TEST(MpsKernelSessionOpsTest, DestroyCommandBufferWithValidBuffer) {
  auto acquired = mps_test_utils::acquireHardware(true, false);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;
  mps_ops::MpsKernelSessionOps::destroyCommandBuffer(hardware.command_buffer);
  hardware.command_buffer = nullptr;
}

TEST(MpsKernelSessionOpsTest, SetBufferWithNullptrEncoderDoesNotCrash) {
  mps_storage::MpsStorage storage;
  mps_ops::MpsKernelSessionOps::setBuffer(nullptr, storage, 0);
}

TEST(MpsKernelSessionOpsTest, SetBufferWithInvalidStorageDoesNotCrash) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  mps_storage::MpsStorage empty_storage;
  mps_ops::MpsKernelSessionOps::setBuffer(hardware.encoder, empty_storage, 0);
}

TEST(MpsKernelSessionOpsTest, SetBytesWithNullptrEncoderDoesNotCrash) {
  int data = 42;
  mps_ops::MpsKernelSessionOps::setBytes(nullptr, &data, sizeof(data), 0);
}

TEST(MpsKernelSessionOpsTest, SetBytesWithNullptrDataDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::setBytes(
      reinterpret_cast<mps_wrapper::MpsComputeCommandEncoder_t>(0x1), nullptr,
      4, 0);
}

TEST(MpsKernelSessionOpsTest, SetBytesWithValidEncoder) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  int data = 42;
  mps_ops::MpsKernelSessionOps::setBytes(hardware.encoder, &data, sizeof(data),
                                         0);
}

TEST(MpsKernelSessionOpsTest, SetParamWithNullptrEncoderDoesNotCrash) {
  auto param =
      ::orteaf::internal::kernel::Param(::orteaf::internal::kernel::ParamId{0},
                                        42);
  mps_ops::MpsKernelSessionOps::setParam(nullptr, param, 0);
}

TEST(MpsKernelSessionOpsTest, SetParamWithIntParam) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  auto param =
      ::orteaf::internal::kernel::Param(::orteaf::internal::kernel::ParamId{0},
                                        42);
  mps_ops::MpsKernelSessionOps::setParam(hardware.encoder, param, 0);
}

TEST(MpsKernelSessionOpsTest, SetParamWithFloatParam) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{1}, 3.14f);
  mps_ops::MpsKernelSessionOps::setParam(hardware.encoder, param, 0);
}

TEST(MpsKernelSessionOpsTest, SetParamWithSizeTParam) {
  auto acquired = mps_test_utils::acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{2}, std::size_t{1024});
  mps_ops::MpsKernelSessionOps::setParam(hardware.encoder, param, 0);
}

TEST(MpsKernelSessionOpsTest, SetPipelineStateWithNullptrEncoderDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::PipelineLease pipeline;
  mps_ops::MpsKernelSessionOps::setPipelineState(nullptr, pipeline);
}

TEST(MpsKernelSessionOpsTest, SetPipelineStateWithEmptyLeaseDoesNotCrash) {
  auto fake_encoder =
      reinterpret_cast<mps_wrapper::MpsComputeCommandEncoder_t>(0x1);
  mps_ops::MpsKernelSessionOps::PipelineLease empty_pipeline;
  mps_ops::MpsKernelSessionOps::setPipelineState(fake_encoder, empty_pipeline);
}

TEST(MpsKernelSessionOpsTest,
     DispatchThreadgroupsWithNullptrEncoderDoesNotCrash) {
  auto threadgroups = mps_wrapper::makeSize(1, 1, 1);
  auto threads_per_threadgroup = mps_wrapper::makeSize(32, 1, 1);
  mps_ops::MpsKernelSessionOps::dispatchThreadgroups(
      nullptr, threadgroups, threads_per_threadgroup);
}

TEST(MpsKernelSessionOpsTest, DispatchThreadsWithNullptrEncoderDoesNotCrash) {
  auto threads_per_grid = mps_wrapper::makeSize(1024, 1, 1);
  auto threads_per_threadgroup = mps_wrapper::makeSize(256, 1, 1);
  mps_ops::MpsKernelSessionOps::dispatchThreads(nullptr, threads_per_grid,
                                                threads_per_threadgroup);
}

TEST(MpsKernelSessionOpsTest, GetGPUTimesWithNullptrReturnsZero) {
  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUStartTime(nullptr), 0.0);
  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUEndTime(nullptr), 0.0);
  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUDuration(nullptr), 0.0);
}

TEST(MpsKernelSessionOpsTest, GetGPUTimesWithUnscheduledBufferReturnsZero) {
  auto acquired = mps_test_utils::acquireHardware(true, false);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUStartTime(hardware.command_buffer),
            0.0);
  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUEndTime(hardware.command_buffer),
            0.0);
  EXPECT_EQ(mps_ops::MpsKernelSessionOps::getGPUDuration(hardware.command_buffer),
            0.0);
}

TEST(MpsKernelSessionOpsTest, MakeGridSizeCreates1DSize) {
  auto size = mps_ops::MpsKernelSessionOps::makeGridSize(256);
  EXPECT_EQ(size.width, 256);
  EXPECT_EQ(size.height, 1);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, MakeGridSizeCreates2DSize) {
  auto size = mps_ops::MpsKernelSessionOps::makeGridSize(16, 16);
  EXPECT_EQ(size.width, 16);
  EXPECT_EQ(size.height, 16);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, MakeGridSizeCreates3DSize) {
  auto size = mps_ops::MpsKernelSessionOps::makeGridSize(8, 8, 8);
  EXPECT_EQ(size.width, 8);
  EXPECT_EQ(size.height, 8);
  EXPECT_EQ(size.depth, 8);
}

TEST(MpsKernelSessionOpsTest, MakeThreadsPerThreadgroupCreates1DSize) {
  auto size = mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(256);
  EXPECT_EQ(size.width, 256);
  EXPECT_EQ(size.height, 1);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, MakeThreadsPerThreadgroupCreates2DSize) {
  auto size = mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(16, 16);
  EXPECT_EQ(size.width, 16);
  EXPECT_EQ(size.height, 16);
  EXPECT_EQ(size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, CalculateGridSizeRoundsUp) {
  auto total_threads = mps_ops::MpsKernelSessionOps::makeGridSize(1000);
  auto threads_per_threadgroup =
      mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(256);
  auto grid_size = mps_ops::MpsKernelSessionOps::calculateGridSize(
      total_threads, threads_per_threadgroup);

  EXPECT_EQ(grid_size.width, 4);
  EXPECT_EQ(grid_size.height, 1);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, CalculateGridSizeExactDivision) {
  auto total_threads = mps_ops::MpsKernelSessionOps::makeGridSize(1024);
  auto threads_per_threadgroup =
      mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(256);
  auto grid_size = mps_ops::MpsKernelSessionOps::calculateGridSize(
      total_threads, threads_per_threadgroup);

  EXPECT_EQ(grid_size.width, 4);
  EXPECT_EQ(grid_size.height, 1);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, CalculateGridSize2D) {
  auto total_threads = mps_ops::MpsKernelSessionOps::makeGridSize(1920, 1080);
  auto threads_per_threadgroup =
      mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(16, 16);
  auto grid_size = mps_ops::MpsKernelSessionOps::calculateGridSize(
      total_threads, threads_per_threadgroup);

  EXPECT_EQ(grid_size.width, 120);
  EXPECT_EQ(grid_size.height, 68);
  EXPECT_EQ(grid_size.depth, 1);
}

TEST(MpsKernelSessionOpsTest, CalculateGridSize3D) {
  auto total_threads = mps_ops::MpsKernelSessionOps::makeGridSize(100, 100, 100);
  auto threads_per_threadgroup =
      mps_ops::MpsKernelSessionOps::makeThreadsPerThreadgroup(8, 8, 8);
  auto grid_size = mps_ops::MpsKernelSessionOps::calculateGridSize(
      total_threads, threads_per_threadgroup);

  EXPECT_EQ(grid_size.width, 13);
  EXPECT_EQ(grid_size.height, 13);
  EXPECT_EQ(grid_size.depth, 13);
}

} // namespace
