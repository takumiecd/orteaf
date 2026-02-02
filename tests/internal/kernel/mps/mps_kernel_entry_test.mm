#include <gtest/gtest.h>

#include <cstddef>
#include <system_error>
#include <variant>

#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>

namespace kernel = orteaf::internal::kernel;
namespace kernel_entry = ::orteaf::internal::kernel::core;

namespace {

TEST(MpsKernelEntryTest, DefaultConstruction) {
  kernel_entry::KernelEntry entry;
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entry.base()));
}

TEST(MpsKernelEntryTest, RunThrowsOnEmptyBase) {
  kernel_entry::KernelEntry entry;
  kernel::KernelArgs args;
  EXPECT_THROW({ entry.run(args); }, std::system_error);
}

} // namespace
