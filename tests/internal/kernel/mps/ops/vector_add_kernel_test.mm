#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <utility>
#include <vector>

#include <orteaf/internal/execution/mps/platform/wrapper/mps_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_buffer.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_compute_command_encoder.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

// Include the kernel under test
#include "tests/internal/kernel/mps/ops/fixtures/vector_add_kernel.h"
#include "tests/internal/kernel/mps/test_utils/mps_hardware_test_utils.h"

namespace kernel = orteaf::internal::kernel;
namespace kernel_entry = ::orteaf::internal::kernel::core;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace mps_resource = ::orteaf::internal::execution::mps::resource;
namespace mps_api = ::orteaf::internal::execution::mps::api;
namespace mps_test_utils =
    ::orteaf::tests::internal::kernel::mps::test_utils;
namespace vector_add = orteaf::extension::kernel::mps::ops;

namespace {

struct MpsExecutionGuard {
  bool ok{false};

  MpsExecutionGuard() {
    mps_api::MpsExecutionApi::ExecutionManager::Config config{};
    const int device_count = mps_wrapper::getDeviceCount();
    const std::size_t capacity =
        device_count <= 0 ? 0u : static_cast<std::size_t>(device_count);
    if (capacity == 0) {
      return;
    }

    auto &device_cfg = config.device_config;
    const std::size_t block_size = capacity > 0 ? capacity : 1;
    device_cfg.control_block_capacity = capacity;
    device_cfg.control_block_block_size = block_size;
    device_cfg.control_block_growth_chunk_size = 1;
    device_cfg.payload_capacity = capacity;
    device_cfg.payload_block_size = block_size;
    device_cfg.payload_growth_chunk_size = 1;

    auto configure_pool = [](auto &cfg, std::size_t pool_capacity) {
      cfg.control_block_capacity = pool_capacity;
      cfg.control_block_block_size = pool_capacity;
      cfg.control_block_growth_chunk_size = 1;
      cfg.payload_capacity = pool_capacity;
      cfg.payload_block_size = pool_capacity;
      cfg.payload_growth_chunk_size = 1;
    };

    configure_pool(device_cfg.command_queue_config, 1);
    configure_pool(device_cfg.event_config, 1);
    configure_pool(device_cfg.fence_config, 1);
    configure_pool(device_cfg.heap_config, 1);
    configure_pool(device_cfg.library_config, 1);
    configure_pool(device_cfg.graph_config, 1);

    configure_pool(config.kernel_base_config, 1);
    configure_pool(config.kernel_metadata_config, 1);
    // FixedSlotStore payloads need slot 0 created via growth on first acquire.
    config.kernel_metadata_config.payload_capacity = 0;

    try {
      mps_api::MpsExecutionApi::configure(config);
      ok = true;
    } catch (...) {
      mps_api::MpsExecutionApi::shutdown();
      ok = false;
    }
  }

  ~MpsExecutionGuard() {
    mps_api::MpsExecutionApi::shutdown();
  }
};

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
  MpsExecutionGuard guard;
  if (!guard.ok) {
    GTEST_SKIP() << "Failed to configure MPS execution";
  }
  auto metadata = vector_add::createVectorAddMetadata();
  auto entry = metadata.rebuild();

  // Check lease variant (MPS lease is expected, may be invalid)
  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  EXPECT_TRUE(std::holds_alternative<MpsLease>(entry.base()));
}

// =============================================================================
// Integration Tests (require real MPS hardware)
// =============================================================================

#if ORTEAF_ENABLE_MPS
class VectorAddKernelIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto acquired = mps_test_utils::acquireHardware(false, false);
    if (!acquired.context) {
      GTEST_SKIP() << acquired.reason;
    }
    hardware_ = std::move(*acquired.context);
  }

  mps_test_utils::MpsHardwareContext hardware_{};
};

TEST_F(VectorAddKernelIntegrationTest, KernelEntryCanBeCreated) {
  MpsExecutionGuard guard;
  if (!guard.ok) {
    GTEST_SKIP() << "Failed to configure MPS execution";
  }
  auto metadata = vector_add::createVectorAddMetadata();
  auto entry = metadata.rebuild();

  using MpsLease = kernel_entry::KernelEntry::MpsKernelBaseLease;
  EXPECT_TRUE(std::holds_alternative<MpsLease>(entry.base()));
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
  // Verify the execute function signature matches the MPS base contract
  using ExpectedFunc = mps_resource::MpsKernelBase::ExecuteFunc;
  auto func = static_cast<ExpectedFunc>(&vector_add::vectorAddExecute);
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

} // namespace
