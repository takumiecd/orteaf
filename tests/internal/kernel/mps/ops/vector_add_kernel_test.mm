#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <system_error>
#include <variant>
#include <vector>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

// Include the kernel under test
#include "tests/internal/kernel/mps/ops/fixtures/vector_add_kernel.h"

namespace kernel = orteaf::internal::kernel;
namespace kernel_entry = ::orteaf::internal::kernel::core;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace vector_add = orteaf::extension::kernel::mps::ops;

namespace {

// =============================================================================
// Schema Tests
// =============================================================================

TEST(VectorAddKernelTest, StorageSchemaHasThreeFields) {
  // Verify the storage schema has the expected structure
  vector_add::VectorAddStorages storages;

  // Check storage IDs
  EXPECT_EQ(storages.a.kId, kernel::OperandId::Input0);
  EXPECT_EQ(storages.b.kId, kernel::OperandId::Input1);
  EXPECT_EQ(storages.c.kId, kernel::OperandId::Output);
}

TEST(VectorAddKernelTest, ParamSchemaHasNumElements) {
  vector_add::VectorAddParams params;

  EXPECT_EQ(params.num_elements.kId, kernel::ParamId::NumElements);
}

TEST(VectorAddKernelTest, CreateKernelEntryHasCorrectStructure) {
  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  auto entry = vector_add::createVectorAddKernel(MpsLease{});

  // Check lease variant (MPS lease is expected, may be invalid)
  EXPECT_TRUE(std::holds_alternative<MpsLease>(entry.base()));
  EXPECT_NE(entry.execute(), nullptr);
}

// =============================================================================
// Integration Tests (require real MPS hardware)
// =============================================================================

#if ORTEAF_ENABLE_MPS
class VectorAddKernelIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    device_ = mps_wrapper::getDevice();
    if (device_ == nullptr) {
      GTEST_SKIP() << "No Metal devices available";
    }

    queue_ = mps_wrapper::createCommandQueue(device_);
    if (queue_ == nullptr) {
      mps_wrapper::deviceRelease(device_);
      GTEST_SKIP() << "Failed to create command queue";
    }
  }

  void TearDown() override {
    if (queue_ != nullptr) {
      mps_wrapper::destroyCommandQueue(queue_);
    }
    if (device_ != nullptr) {
      mps_wrapper::deviceRelease(device_);
    }
  }

  mps_wrapper::MpsDevice_t device_{nullptr};
  mps_wrapper::MpsCommandQueue_t queue_{nullptr};
};

TEST_F(VectorAddKernelIntegrationTest, KernelEntryCanBeCreated) {
  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  auto entry = vector_add::createVectorAddKernel(MpsLease{});

  EXPECT_TRUE(std::holds_alternative<MpsLease>(entry.base()));
  EXPECT_NE(entry.execute(), nullptr);
}

// NOTE: Full integration tests would require:
// 1. Setting up a proper execution context with device/queue leases
// 2. Creating MpsStorage instances for input/output buffers
// 3. Registering the Metal library with the library manager
// 4. Calling entry.run() and verifying results
//
// These tests are placeholder for demonstrating the API usage.
// Full integration would require the complete runtime setup.

TEST_F(VectorAddKernelIntegrationTest, ExecuteFunctionSignatureIsCorrect) {
  // Verify the execute function signature matches what KernelEntry expects
  using ExpectedFunc =
      void (*)(kernel_entry::KernelEntry::KernelBaseLease &, kernel::KernelArgs &);

  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  auto entry = vector_add::createVectorAddKernel(MpsLease{});

  // This static_cast will fail at compile time if signature is wrong
  ExpectedFunc func = entry.execute();
  EXPECT_NE(func, nullptr);
}

#endif // ORTEAF_ENABLE_MPS

// =============================================================================
// Execute Function Logic Tests (without actual GPU dispatch)
// =============================================================================

TEST(VectorAddKernelTest, KernelArgsDefaultContextIsInvalid) {
  kernel::KernelArgs args;
  EXPECT_FALSE(args.valid());
}

TEST(VectorAddKernelTest, ExecuteFunctionThrowsWhenStoragesNotBound) {
  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  auto entry = vector_add::createVectorAddKernel(MpsLease{});
  kernel::KernelArgs args;

  // Execute throws because required storage bindings are not set
  auto func = entry.execute();
  ASSERT_NE(func, nullptr);
  EXPECT_THROW({ func(entry.base(), args); }, std::system_error);
}

} // namespace
