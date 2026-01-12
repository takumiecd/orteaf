#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/diagnostics/log/log_config.h>
#include <orteaf/internal/execution/mps/manager/mps_command_queue_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_fence_manager.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_queue.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_event.h>
#include <tests/internal/execution/mps/manager/testing/execution_ops_provider.h>
#include <tests/internal/execution/mps/manager/testing/manager_test_fixture.h>
#include <tests/internal/testing/error_assert.h>

namespace diag_error = orteaf::internal::diagnostics::error;
namespace mps_rt = orteaf::internal::execution::mps::manager;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace testing_mps = orteaf::tests::execution::mps::testing;

using orteaf::tests::ExpectError;

namespace {

mps_wrapper::MpsCommandQueue_t makeQueue(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsCommandQueue_t>(value);
}

mps_rt::MpsCommandQueueManager::Config
makeConfig(std::size_t payload_capacity, std::size_t control_block_capacity,
           std::size_t payload_block_size, std::size_t control_block_block_size,
           std::size_t payload_growth_chunk_size,
           std::size_t control_block_growth_chunk_size) {
  mps_rt::MpsCommandQueueManager::Config config{};
  config.payload_capacity = payload_capacity;
  config.control_block_capacity = control_block_capacity;
  config.payload_block_size = payload_block_size;
  config.control_block_block_size = control_block_block_size;
  config.payload_growth_chunk_size = payload_growth_chunk_size;
  config.control_block_growth_chunk_size = control_block_growth_chunk_size;
  return config;
}

template <class Provider>
class MpsCommandQueueManagerTypedTest
    : public testing_mps::RuntimeManagerFixture<
          Provider, mps_rt::MpsCommandQueueManager> {
protected:
  using Base =
      testing_mps::RuntimeManagerFixture<Provider,
                                         mps_rt::MpsCommandQueueManager>;

  mps_rt::MpsCommandQueueManager &manager() { return Base::manager(); }
  auto &adapter() { return Base::adapter(); }
};

#if ORTEAF_ENABLE_MPS
using ProviderTypes = ::testing::Types<testing_mps::MockExecutionOpsProvider,
                                       testing_mps::RealExecutionOpsProvider>;
#else
using ProviderTypes = ::testing::Types<testing_mps::MockExecutionOpsProvider>;
#endif

} // namespace

TYPED_TEST_SUITE(MpsCommandQueueManagerTypedTest, ProviderTypes);

// =============================================================================
// Configuration Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, GrowthChunkSizeCanBeAdjusted) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Assert: Default is 1
  EXPECT_EQ(manager.payloadGrowthChunkSizeForTest(), 1u);
  EXPECT_EQ(manager.controlBlockGrowthChunkSizeForTest(), 1u);

  this->adapter().expectCreateCommandQueues({makeQueue(0x050)});
  this->adapter().expectDestroyCommandQueues({makeQueue(0x050)});

  // Act
  manager.configureForTest(makeConfig(1, 1, 1, 1, 4, 5), device,
                           this->getOps());

  // Assert
  EXPECT_EQ(manager.payloadGrowthChunkSizeForTest(), 4u);
  EXPECT_EQ(manager.controlBlockGrowthChunkSizeForTest(), 5u);

  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, GrowthChunkSizeRejectsZero) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Act & Assert
  ExpectError(diag_error::OrteafErrc::InvalidArgument, [&] {
    manager.configureForTest(makeConfig(1, 1, 1, 1, 0, 1), device,
                             this->getOps());
  });
  ExpectError(diag_error::OrteafErrc::InvalidArgument, [&] {
    manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 0), device,
                             this->getOps());
  });
}

// =============================================================================
// Initialization Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, InitializeSetsCapacity) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x100), makeQueue(0x101)});
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x100), makeQueue(0x101)});

  // Act
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1), device,
                           this->getOps());

  // Assert
  EXPECT_EQ(manager.payloadPoolSizeForTest(), 2u);
  EXPECT_TRUE(manager.isConfiguredForTest());

  // Cleanup
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, CapacityReflectsPoolSize) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Before init
  EXPECT_EQ(manager.payloadPoolSizeForTest(), 0u);

  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x200), makeQueue(0x201), makeQueue(0x202)});
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x200), makeQueue(0x201), makeQueue(0x202)});

  // Act
  manager.configureForTest(makeConfig(3, 3, 3, 3, 1, 1), device,
                           this->getOps());

  // Assert
  EXPECT_EQ(manager.payloadPoolSizeForTest(), 3u);

  // Cleanup
  manager.shutdown();
  EXPECT_EQ(manager.payloadPoolSizeForTest(), 0u);
}

// =============================================================================
// Acquire Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, AcquireFailsBeforeInitialization) {
  auto &manager = this->manager();

  // Act & Assert
  ExpectError(diag_error::OrteafErrc::InvalidState,
              [&] { (void)manager.acquire(); });
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           AcquireCreatesQueueAndReturnsLease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x300), makeQueue(0x301)});
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x300), makeQueue(0x301)});
  manager.configureForTest(makeConfig(2, 2, 2, 2, 1, 1), device,
                           this->getOps());

  {
    // Act
    auto lease = manager.acquire();

    // Assert
    auto *payload = lease.operator->();
    ASSERT_NE(payload, nullptr);
    EXPECT_TRUE(payload->hasQueue());
  } // lease released here

  // Cleanup
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           AcquireBindsLifetimeManagerForQueueHandle) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  mps_rt::MpsFenceManager fence_manager{};
  mps_rt::MpsFenceManager::Config fence_config{};
  fence_config.payload_capacity = 0;
  fence_config.control_block_capacity = 0;
  fence_config.payload_block_size = 1;
  fence_config.control_block_block_size = 1;
  fence_config.payload_growth_chunk_size = 1;
  fence_config.control_block_growth_chunk_size = 1;
  fence_manager.configureForTest(fence_config, device, this->getOps());

  this->adapter().expectCreateCommandQueues({makeQueue(0x305)});
  this->adapter().expectDestroyCommandQueues({makeQueue(0x305)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps(), &fence_manager);

  auto lease = manager.acquire();
  auto *payload = lease.operator->();
  ASSERT_NE(payload, nullptr);
  EXPECT_TRUE(payload->hasQueue());
  EXPECT_NO_THROW((void)payload->lifetime().releaseReady());

  lease.release();
  manager.shutdown();
  fence_manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, AcquireGrowsPoolWhenNeeded) {
  printf("AcquireGrowsPoolWhenNeeded 0\n");
  auto &manager = this->manager();
  const auto device = this->adapter().device();
  printf("AcquireGrowsPoolWhenNeeded\n");
  // Arrange - start with capacity 1
  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x400), makeQueue(0x401)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  printf("AcquireGrowsPoolWhenNeeded 1\n");

  {
    printf("AcquireGrowsPoolWhenNeeded 2\n");
    // First acquire creates first queue
    auto first = manager.acquire();

    // Second acquire triggers growth and creates second queue
    auto second = manager.acquire();

    printf("AcquireGrowsPoolWhenNeeded 3\n");

    // Assert
    EXPECT_NE(first.payloadHandle(), second.payloadHandle());
  } // leases released here

  printf("AcquireGrowsPoolWhenNeeded 4\n");

  // Cleanup
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x400), makeQueue(0x401)});

  manager.shutdown();
}

// =============================================================================
// Acquire by Handle Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, AcquireByHandleReturnsLease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x500)});
  this->adapter().expectDestroyCommandQueues({makeQueue(0x500)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  mps_rt::MpsCommandQueueManager::CommandQueueHandle handle{};
  {
    auto original = manager.acquire();
    handle = original.payloadHandle();

    // Act: Acquire by handle creates new control block but same payload
    auto byHandle = manager.acquire(handle);

    // Assert: Same payload handle
    EXPECT_EQ(byHandle.payloadHandle(), handle);
    // payloadPtr may differ because different control blocks point to same slot
  } // leases released here

  // Cleanup
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, AcquireByInvalidHandleFails) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x730)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  // Act & Assert
  using Handle = mps_rt::MpsCommandQueueManager::CommandQueueHandle;
  ExpectError(diag_error::OrteafErrc::InvalidArgument,
              [&] { (void)manager.acquire(Handle::invalid()); });

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x730)});
  manager.shutdown();
}

// =============================================================================
// Weak Lease Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, LeaseDestructionAllowsShutdown) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x500)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  // Act: Lease goes out of scope
  {
    auto lease = manager.acquire();
    auto *payload = lease.operator->();
    ASSERT_NE(payload, nullptr);
    EXPECT_TRUE(payload->hasQueue());
  } // lease released here

  // Assert: Can still shutdown (weak lease released before shutdown)
  this->adapter().expectDestroyCommandQueues({makeQueue(0x500)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, LeaseCopyIncrementsStrongCount) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x510)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  {
    // Act
    auto lease1 = manager.acquire();
    auto initial_count = lease1.strongCount();
    auto lease2 = lease1; // Copy

    // Assert
    EXPECT_EQ(lease2.strongCount(), initial_count + 1);
  } // leases released here

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x510)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, LeaseMoveDoesNotChangeStrongCount) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x520)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  {
    // Act
    auto lease1 = manager.acquire();
    auto initial_count = lease1.strongCount();
    auto lease2 = std::move(lease1); // Move

    // Assert
    EXPECT_EQ(lease2.strongCount(), initial_count);
  } // lease2 released here

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x520)});
  manager.shutdown();
}

// =============================================================================
// isAlive Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, IsAliveReturnsTrueForValidHandle) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x700)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  mps_rt::MpsCommandQueueManager::CommandQueueHandle handle{};
  {
    auto lease = manager.acquire();
    handle = lease.payloadHandle();

    // Assert
    EXPECT_TRUE(manager.isAliveForTest(handle));
  } // lease released here

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x700)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           IsAliveReturnsFalseForInvalidHandle) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x710)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  // Assert
  using Handle = mps_rt::MpsCommandQueueManager::CommandQueueHandle;
  EXPECT_FALSE(manager.isAliveForTest(Handle::invalid()));

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x710)});
  manager.shutdown();
}

#if ORTEAF_ENABLE_TEST
TYPED_TEST(MpsCommandQueueManagerTypedTest, ControlBlockPoolCapacityForTest) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x720)});
  manager.configureForTest(makeConfig(1, 1, 1, 1, 1, 1), device,
                           this->getOps());

  // Assert
  EXPECT_GE(manager.controlBlockPoolCapacityForTest(), 1u);

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x720)});
  manager.shutdown();
}
#endif
