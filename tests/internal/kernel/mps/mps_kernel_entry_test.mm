#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <variant>

#include <orteaf/internal/execution/mps/manager/mps_compute_pipeline_state_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/kernel_entry.h>

namespace kernel = orteaf::internal::kernel;
namespace kernel_entry = ::orteaf::internal::kernel;

namespace {

// Test helper: track execution calls
static bool g_execute_called = false;
static kernel_entry::KernelEntry::KernelBaseLease *g_execute_lease = nullptr;
static kernel::KernelArgs *g_execute_args = nullptr;

void mockExecuteFunc(kernel_entry::KernelEntry::KernelBaseLease &lease,
                     kernel::KernelArgs &args) {
  g_execute_called = true;
  g_execute_lease = &lease;
  g_execute_args = &args;
}

void resetExecuteTracking() {
  g_execute_called = false;
  g_execute_lease = nullptr;
  g_execute_args = nullptr;
}

TEST(MpsKernelEntryTest, DefaultConstruction) {
  kernel_entry::KernelEntry entry;
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entry.base()));
  EXPECT_EQ(entry.execute(), nullptr);
}

TEST(MpsKernelEntryTest, SetExecuteFunction) {
  kernel_entry::KernelEntry entry;
  entry.setExecute(mockExecuteFunc);
  EXPECT_NE(entry.execute(), nullptr);
}

TEST(MpsKernelEntryTest, RunThrowsOnEmptyBase) {
  resetExecuteTracking();

  kernel_entry::KernelEntry entry;
  entry.setExecute(mockExecuteFunc);

  kernel::KernelArgs args;
  EXPECT_THROW({ entry.run(args); }, std::runtime_error);
}

} // namespace
