#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <system_error>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_size.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session_ops.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/storage/mps/mps_storage.h>

namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_ops = ::orteaf::internal::kernel::mps;
namespace mps_wrapper = ::orteaf::internal::execution::mps::platform::wrapper;
namespace mps_storage = ::orteaf::internal::storage::mps;

namespace {

struct MpsHardwareContext {
  mps_wrapper::MpsDevice_t device{nullptr};
  mps_wrapper::MpsCommandQueue_t queue{nullptr};
  mps_wrapper::MpsCommandBuffer_t command_buffer{nullptr};
  mps_wrapper::MpsComputeCommandEncoder_t encoder{nullptr};

  MpsHardwareContext() = default;
  MpsHardwareContext(MpsHardwareContext &&other) noexcept
      : device(other.device), queue(other.queue),
        command_buffer(other.command_buffer), encoder(other.encoder) {
    other.device = nullptr;
    other.queue = nullptr;
    other.command_buffer = nullptr;
    other.encoder = nullptr;
  }
  MpsHardwareContext &operator=(MpsHardwareContext &&other) noexcept {
    if (this != &other) {
      cleanup();
      device = other.device;
      queue = other.queue;
      command_buffer = other.command_buffer;
      encoder = other.encoder;
      other.device = nullptr;
      other.queue = nullptr;
      other.command_buffer = nullptr;
      other.encoder = nullptr;
    }
    return *this;
  }
  MpsHardwareContext(const MpsHardwareContext &) = delete;
  MpsHardwareContext &operator=(const MpsHardwareContext &) = delete;
  ~MpsHardwareContext() { cleanup(); }

private:
  void cleanup() {
    if (encoder != nullptr) {
      mps_wrapper::destroyComputeCommandEncoder(encoder);
      encoder = nullptr;
    }
    if (command_buffer != nullptr) {
      mps_wrapper::destroyCommandBuffer(command_buffer);
      command_buffer = nullptr;
    }
    if (queue != nullptr) {
      mps_wrapper::destroyCommandQueue(queue);
      queue = nullptr;
    }
    if (device != nullptr) {
      mps_wrapper::deviceRelease(device);
      device = nullptr;
    }
  }
};

struct HardwareAcquireResult {
  std::optional<MpsHardwareContext> context;
  std::string reason;
};

HardwareAcquireResult acquireHardware(bool need_buffer = false,
                                      bool need_encoder = false) {
  HardwareAcquireResult result{};
  MpsHardwareContext hardware{};

  try {
    hardware.device = mps_wrapper::getDevice();
  } catch (const std::system_error &err) {
    result.reason = err.what();
    return result;
  }

  if (hardware.device == nullptr) {
    result.reason = "No Metal devices available";
    return result;
  }

  hardware.queue = mps_wrapper::createCommandQueue(hardware.device);
  if (hardware.queue == nullptr) {
    result.reason = "Failed to create command queue";
    return result;
  }

  if (need_buffer || need_encoder) {
    hardware.command_buffer = mps_wrapper::createCommandBuffer(hardware.queue);
    if (hardware.command_buffer == nullptr) {
      result.reason = "Failed to create command buffer";
      return result;
    }
  }

  if (need_encoder) {
    hardware.encoder =
        mps_wrapper::createComputeCommandEncoder(hardware.command_buffer);
    if (hardware.encoder == nullptr) {
      result.reason = "Failed to create compute command encoder";
      return result;
    }
  }

  result.context = std::move(hardware);
  return result;
}

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
  auto acquired = acquireHardware(true, false);
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
  auto acquired = acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;
  mps_ops::MpsKernelSessionOps::endEncoding(hardware.encoder);
}

TEST(MpsKernelSessionOpsTest, CommitWithNullptrBufferDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::commit(nullptr);
}

TEST(MpsKernelSessionOpsTest, WaitUntilCompletedWithNullptrBufferDoesNotCrash) {
  mps_ops::MpsKernelSessionOps::waitUntilCompleted(nullptr);
}

TEST(MpsKernelSessionOpsTest, CommitAndWaitWithValidBuffer) {
  auto acquired = acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  mps_ops::MpsKernelSessionOps::endEncoding(hardware.encoder);
  mps_ops::MpsKernelSessionOps::commit(hardware.command_buffer);
  mps_ops::MpsKernelSessionOps::waitUntilCompleted(hardware.command_buffer);
}

TEST(MpsKernelSessionOpsTest, SetBufferWithNullptrEncoderDoesNotCrash) {
  mps_storage::MpsStorage storage;
  mps_ops::MpsKernelSessionOps::setBuffer(nullptr, storage, 0);
}

TEST(MpsKernelSessionOpsTest, SetBufferWithInvalidStorageDoesNotCrash) {
  auto acquired = acquireHardware(true, true);
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
  auto acquired = acquireHardware(true, true);
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
  auto acquired = acquireHardware(true, true);
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
  auto acquired = acquireHardware(true, true);
  if (!acquired.context) {
    GTEST_SKIP() << acquired.reason;
  }
  auto &hardware = *acquired.context;

  auto param = ::orteaf::internal::kernel::Param(
      ::orteaf::internal::kernel::ParamId{1}, 3.14f);
  mps_ops::MpsKernelSessionOps::setParam(hardware.encoder, param, 0);
}

TEST(MpsKernelSessionOpsTest, SetParamWithSizeTParam) {
  auto acquired = acquireHardware(true, true);
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
  auto acquired = acquireHardware(true, false);
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
