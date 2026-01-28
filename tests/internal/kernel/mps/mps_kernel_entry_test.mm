#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>

#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_entry.h>

namespace kernel = orteaf::internal::kernel;
namespace mps_kernel = orteaf::internal::kernel::mps;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;
namespace mps_context = orteaf::internal::execution_context::mps;

namespace {

// Mock RuntimeApi for testing configure()
struct MockRuntimeApi {
  using PipelineLease = mps_kernel::MpsKernelBase::PipelineLease;
  using LibraryKey = mps_kernel::MpsKernelBase::LibraryKey;
  using FunctionKey = mps_kernel::MpsKernelBase::FunctionKey;

  static PipelineLease acquirePipeline(mps_exec::MpsDeviceHandle device,
                                       const LibraryKey &lib,
                                       const FunctionKey &func) {
    return PipelineLease{};
  }
};

// Test helper: track execution calls
static bool g_execute_called = false;
static mps_kernel::MpsKernelBase *g_execute_base = nullptr;
static kernel::KernelArgs *g_execute_args = nullptr;

void mockExecuteFunc(mps_kernel::MpsKernelBase &base,
                     kernel::KernelArgs &args) {
  g_execute_called = true;
  g_execute_base = &base;
  g_execute_args = &args;
}

void resetExecuteTracking() {
  g_execute_called = false;
  g_execute_base = nullptr;
  g_execute_args = nullptr;
}

TEST(MpsKernelEntryTest, DefaultConstruction) {
  mps_kernel::MpsKernelEntry entry;
  EXPECT_EQ(entry.base.kernelCount(), 0u);
  EXPECT_EQ(entry.execute, nullptr);
}

TEST(MpsKernelEntryTest, SetExecuteFunction) {
  mps_kernel::MpsKernelEntry entry;
  entry.execute = mockExecuteFunc;
  EXPECT_NE(entry.execute, nullptr);
}

TEST(MpsKernelEntryTest, BaseCanBeConfigured) {
  mps_kernel::MpsKernelEntry entry;
  entry.base.addKey("lib1", "func1");

  EXPECT_EQ(entry.base.kernelCount(), 1u);
}

TEST(MpsKernelEntryTest, RunWithMultipleKernels) {
  resetExecuteTracking();

  mps_kernel::MpsKernelEntry entry;
  entry.base.addKey("lib1", "func1");
  entry.base.addKey("lib2", "func2");
  entry.base.addKey("lib3", "func3");
  entry.execute = mockExecuteFunc;

  EXPECT_EQ(entry.base.kernelCount(), 3u);
}

} // namespace
