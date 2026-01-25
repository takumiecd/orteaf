#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_entry.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/storage_id.h>

// Include the kernel under test
#include "orteaf/src/extension/kernel/mps/ops/vector_add_kernel.h"

namespace kernel = orteaf::internal::kernel;
namespace mps_kernel = orteaf::internal::kernel::mps;
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
  EXPECT_EQ(storages.a.kId, kernel::StorageId::Input0);
  EXPECT_EQ(storages.b.kId, kernel::StorageId::Input1);
  EXPECT_EQ(storages.c.kId, kernel::StorageId::Output);
}

TEST(VectorAddKernelTest, ParamSchemaHasNumElements) {
  vector_add::VectorAddParams params;

  EXPECT_EQ(params.num_elements.kId, kernel::ParamId::NumElements);
}

TEST(VectorAddKernelTest, CreateKernelEntryHasCorrectStructure) {
  auto entry = vector_add::createVectorAddKernel();

  // Check kernel count (should have 1 kernel registered)
  EXPECT_EQ(entry.base.kernelCount(), 1u);

  // Check execute function is set
  EXPECT_NE(entry.execute, nullptr);
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
  auto entry = vector_add::createVectorAddKernel();

  EXPECT_EQ(entry.base.kernelCount(), 1u);
  EXPECT_NE(entry.execute, nullptr);
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
  // Verify the execute function signature matches what MpsKernelEntry expects
  using ExpectedFunc =
      void (*)(mps_kernel::MpsKernelBase &, mps_kernel::MpsKernelArgs &);

  auto entry = vector_add::createVectorAddKernel();

  // This static_cast will fail at compile time if signature is wrong
  ExpectedFunc func = entry.execute;
  EXPECT_NE(func, nullptr);
}

#endif // ORTEAF_ENABLE_MPS

// =============================================================================
// Execute Function Logic Tests (without actual GPU dispatch)
// =============================================================================

TEST(VectorAddKernelTest, ExecuteFunctionThrowsWithoutConfiguredRuntime) {
  // Default constructor requires MPS runtime to be configured
  EXPECT_THROW({ mps_kernel::MpsKernelArgs args; }, std::runtime_error);
}

TEST(VectorAddKernelTest, NoInitAllowsConstructionWithoutRuntime) {
  // NoInit allows construction without MPS runtime (for testing)
  mps_kernel::MpsKernelArgs args{mps_kernel::MpsKernelArgs::NoInit{}};

  // Context has null leases, but construction succeeds
  EXPECT_FALSE(args.context().device);
  EXPECT_FALSE(args.context().command_queue);
}

TEST(VectorAddKernelTest, ExecuteFunctionThrowsWhenStoragesNotBound) {
  auto entry = vector_add::createVectorAddKernel();
  mps_kernel::MpsKernelArgs args{mps_kernel::MpsKernelArgs::NoInit{}};

  // Execute throws because required storage bindings are not set
  EXPECT_THROW({ entry.execute(entry.base, args); }, std::runtime_error);
}

} // namespace
