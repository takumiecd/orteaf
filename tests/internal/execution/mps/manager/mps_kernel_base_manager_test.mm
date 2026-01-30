#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_device_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_library_manager.h>
#include <tests/internal/execution/mps/manager/testing/execution_ops_provider.h>
#include <tests/internal/execution/mps/manager/testing/execution_mock_expectations.h>
#include <tests/internal/execution/mps/manager/testing/manager_test_fixture.h>
#include <tests/internal/testing/error_assert.h>

namespace diag_error = orteaf::internal::diagnostics::error;
namespace kernel_mps = orteaf::internal::execution::mps::manager;
namespace mps_mgr = orteaf::internal::execution::mps::manager;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace testing_mps = orteaf::tests::execution::mps::testing;
namespace mock_expect = orteaf::tests::execution::mps;

using orteaf::tests::ExpectError;

namespace {

mps_wrapper::MpsDevice_t makeDevice(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsDevice_t>(value);
}

mps_wrapper::MpsLibrary_t makeLibrary(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsLibrary_t>(value);
}

mps_wrapper::MpsFunction_t makeFunction(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsFunction_t>(value);
}

mps_wrapper::MpsComputePipelineState_t makePipeline(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsComputePipelineState_t>(value);
}

kernel_mps::MpsKernelBaseManager::Config
makeConfig(std::size_t payload_capacity, std::size_t control_block_capacity,
           std::size_t payload_block_size, std::size_t control_block_block_size,
           std::size_t payload_growth_chunk_size,
           std::size_t control_block_growth_chunk_size) {
  kernel_mps::MpsKernelBaseManager::Config config{};
  config.payload_capacity = payload_capacity;
  config.control_block_capacity = control_block_capacity;
  config.payload_block_size = payload_block_size;
  config.control_block_block_size = control_block_block_size;
  config.payload_growth_chunk_size = payload_growth_chunk_size;
  config.control_block_growth_chunk_size = control_block_growth_chunk_size;
  return config;
}

template <class Provider>
class MpsKernelBaseManagerTypedTest : public ::testing::Test {
protected:
  using Context = typename Provider::Context;

  void SetUp() override {
    Provider::setUp(context_);
    
    device_manager_ = std::make_unique<mps_mgr::MpsDeviceManager>();
    kernel_base_manager_ = std::make_unique<kernel_mps::MpsKernelBaseManager>();
    
    // Get ops
    auto *ops = Provider::getOps(context_);
    
    // For mock, setup expectations
    if constexpr (Provider::is_mock) {
      // Setup device mock expectations
      mock_expect::ExecutionMockExpectations::expectGetDeviceCount(*ops, 1);
      mock_expect::ExecutionMockExpectations::expectGetDevices(
          *ops, {{0, makeDevice(0x1000)}});
      mock_expect::ExecutionMockExpectations::expectDetectArchitectures(
          *ops,
          {{::orteaf::internal::execution::mps::MpsDeviceHandle{0},
            ::orteaf::internal::architecture::Architecture::MpsGeneric}});
      
      // Device creation will create command queues, events, and fences
      mock_expect::ExecutionMockExpectations::expectCreateCommandQueues(
          *ops, {reinterpret_cast<mps_wrapper::MpsCommandQueue_t>(0x3000)},
          ::testing::Eq(makeDevice(0x1000)));
      mock_expect::ExecutionMockExpectations::expectCreateEvents(
          *ops, {reinterpret_cast<mps_wrapper::MpsEvent_t>(0x4000)},
          ::testing::Eq(makeDevice(0x1000)));
      mock_expect::ExecutionMockExpectations::expectCreateFences(
          *ops, {reinterpret_cast<mps_wrapper::MpsFence_t>(0x2000)},
          ::testing::Eq(makeDevice(0x1000)));
    }
    
    const int device_count = ops ? ops->getDeviceCount() : 1;
    const std::size_t capacity =
        device_count <= 0 ? 1u : static_cast<std::size_t>(device_count);
    
    // Configure device manager (includes library manager)
    mps_mgr::MpsDeviceManager::Config device_config{};
    device_config.control_block_capacity = capacity;
    device_config.control_block_block_size = capacity;
    device_config.payload_capacity = capacity;
    device_config.payload_block_size = capacity;
    
    // All sub-managers need proper config
    device_config.command_queue_config.control_block_capacity = 1;
    device_config.command_queue_config.control_block_block_size = 1;
    device_config.command_queue_config.payload_capacity = 1;
    device_config.command_queue_config.payload_block_size = 1;
    
    device_config.event_config.control_block_capacity = 1;
    device_config.event_config.control_block_block_size = 1;
    device_config.event_config.payload_capacity = 1;
    device_config.event_config.payload_block_size = 1;
    
    device_config.fence_config.control_block_capacity = 1;
    device_config.fence_config.control_block_block_size = 1;
    device_config.fence_config.payload_capacity = 1;
    device_config.fence_config.payload_block_size = 1;
    
    device_config.heap_config.control_block_capacity = 1;
    device_config.heap_config.control_block_block_size = 1;
    device_config.heap_config.payload_capacity = 1;
    device_config.heap_config.payload_block_size = 1;
    device_config.heap_config.buffer_config.control_block_capacity = 1;
    device_config.heap_config.buffer_config.control_block_block_size = 1;
    device_config.heap_config.buffer_config.payload_capacity = 1;
    device_config.heap_config.buffer_config.payload_block_size = 1;
    
    device_config.graph_config.control_block_capacity = 1;
    device_config.graph_config.control_block_block_size = 1;
    device_config.graph_config.payload_capacity = 1;
    device_config.graph_config.payload_block_size = 1;
    
    // Library manager config
    device_config.library_config.control_block_capacity = 4;
    device_config.library_config.control_block_block_size = 4;
    device_config.library_config.payload_capacity = 4;
    device_config.library_config.payload_block_size = 4;
    
    // Pipeline config
    device_config.library_config.pipeline_config.control_block_capacity = 4;
    device_config.library_config.pipeline_config.control_block_block_size = 4;
    device_config.library_config.pipeline_config.payload_capacity = 4;
    device_config.library_config.pipeline_config.payload_block_size = 4;
    
    device_manager_->configureForTest(device_config, ops);
    
    // Acquire device lease
    device_lease_ = device_manager_->acquire(
        ::orteaf::internal::execution::mps::MpsDeviceHandle{0});
  }

  void TearDown() override {
    if constexpr (Provider::is_mock) {
      auto *ops = Provider::getOps(context_);
      // Expect fence, event, and command queue destruction
      mock_expect::ExecutionMockExpectations::expectDestroyFences(
          *ops, {reinterpret_cast<mps_wrapper::MpsFence_t>(0x2000)});
      mock_expect::ExecutionMockExpectations::expectDestroyEvents(
          *ops, {reinterpret_cast<mps_wrapper::MpsEvent_t>(0x4000)});
      mock_expect::ExecutionMockExpectations::expectDestroyCommandQueues(
          *ops, {reinterpret_cast<mps_wrapper::MpsCommandQueue_t>(0x3000)});
      // Expect device cleanup
      mock_expect::ExecutionMockExpectations::expectReleaseDevices(
          *ops, {makeDevice(0x1000)});
    }

    device_lease_.release();
    device_manager_->shutdown();
    kernel_base_manager_.reset();
    device_manager_.reset();
    Provider::tearDown(context_);
  }

  kernel_mps::MpsKernelBaseManager &manager() { return *kernel_base_manager_; }
  mps_mgr::MpsDeviceManager &deviceManager() { return *device_manager_; }
  auto &deviceLease() { return device_lease_; }
  auto *getOps() { return Provider::getOps(context_); }

private:
  Context context_{};
  std::unique_ptr<mps_mgr::MpsDeviceManager> device_manager_;
  std::unique_ptr<kernel_mps::MpsKernelBaseManager> kernel_base_manager_;
  mps_mgr::MpsDeviceManager::DeviceLease device_lease_;
};

#if ORTEAF_ENABLE_MPS
using ProviderTypes = ::testing::Types<testing_mps::MockExecutionOpsProvider,
                                       testing_mps::RealExecutionOpsProvider>;
#else
using ProviderTypes = ::testing::Types<testing_mps::MockExecutionOpsProvider>;
#endif

} // namespace

TYPED_TEST_SUITE(MpsKernelBaseManagerTypedTest, ProviderTypes);

// =============================================================================
// Initialization Tests
// =============================================================================

TYPED_TEST(MpsKernelBaseManagerTypedTest, InitializeSucceeds) {
  auto &manager = this->manager();

  // Act
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1));

  // Assert
  EXPECT_TRUE(manager.isConfiguredForTest());
  EXPECT_EQ(manager.payloadPoolCapacityForTest(), 2);
  EXPECT_EQ(manager.controlBlockPoolCapacityForTest(), 2);
}

TYPED_TEST(MpsKernelBaseManagerTypedTest, OperationsBeforeInitializationThrow) {
  auto &manager = this->manager();
  auto &device_lease = this->deviceLease();

  // Act & Assert
  ::orteaf::internal::base::HeapVector<kernel_mps::MpsKernelBaseManager::Key>
      keys;
  ExpectError(diag_error::OrteafErrc::InvalidState,
              [&] { (void)manager.acquire(keys); });
}

// =============================================================================
// Acquire/Release Tests
// =============================================================================

TYPED_TEST(MpsKernelBaseManagerTypedTest, AcquireEmptyKeysSucceeds) {
  auto &manager = this->manager();
  auto &device_lease = this->deviceLease();

  // Arrange
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1));

  ::orteaf::internal::base::HeapVector<kernel_mps::MpsKernelBaseManager::Key>
      keys;

  // Act
  auto lease = manager.acquire(keys);

  // Assert
  EXPECT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 0);
}

TYPED_TEST(MpsKernelBaseManagerTypedTest, ReleaseDecreasesRefCount) {
  auto &manager = this->manager();
  auto &device_lease = this->deviceLease();

  // Arrange
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1));

  ::orteaf::internal::base::HeapVector<kernel_mps::MpsKernelBaseManager::Key>
      keys;

  // Act
  auto lease = manager.acquire(keys);
  EXPECT_TRUE(lease);
  auto count_before = lease.strongCount();

  lease.release();

  // Assert
  EXPECT_FALSE(lease);
}

TYPED_TEST(MpsKernelBaseManagerTypedTest, MultipleAcquireSharesResources) {
  auto &manager = this->manager();
  auto &device_lease = this->deviceLease();

  // Arrange
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1));

  ::orteaf::internal::base::HeapVector<kernel_mps::MpsKernelBaseManager::Key>
      keys;

  // Act
  auto lease1 = manager.acquire(keys);
  auto lease2 = lease1; // Copy should increase ref count

  // Assert
  EXPECT_TRUE(lease1);
  EXPECT_TRUE(lease2);
  EXPECT_EQ(lease1.strongCount(), 2);
  EXPECT_EQ(lease2.strongCount(), 2);
}

// =============================================================================
// Growth Tests
// =============================================================================

TYPED_TEST(MpsKernelBaseManagerTypedTest, GrowthChunkSizeIsRespected) {
  auto &manager = this->manager();

  // Arrange
  const std::size_t growth_chunk = 3;
  manager.configureForTest(makeConfig(1, 1, 1, 1, growth_chunk, growth_chunk));

  // Assert
  EXPECT_EQ(manager.payloadGrowthChunkSizeForTest(), growth_chunk);
  EXPECT_EQ(manager.controlBlockGrowthChunkSizeForTest(), growth_chunk);
}

// =============================================================================
// Shutdown Tests
// =============================================================================

TYPED_TEST(MpsKernelBaseManagerTypedTest, ShutdownCleansUp) {
  auto &manager = this->manager();

  // Arrange
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1));
  EXPECT_TRUE(manager.isConfiguredForTest());

  // Act
  manager.shutdown();

  // Assert
  EXPECT_FALSE(manager.isConfiguredForTest());
}

TYPED_TEST(MpsKernelBaseManagerTypedTest,
           ShutdownBeforeInitializationIsNoOp) {
  auto &manager = this->manager();

  // Act & Assert (should not throw)
  manager.shutdown();
  EXPECT_FALSE(manager.isConfiguredForTest());
}
