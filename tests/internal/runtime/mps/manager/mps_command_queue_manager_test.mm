#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/diagnostics/log/log_config.h>
#include <orteaf/internal/runtime/mps/manager/mps_command_queue_manager.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_command_queue.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_event.h>
#include <tests/internal/runtime/mps/manager/testing/backend_ops_provider.h>
#include <tests/internal/runtime/mps/manager/testing/manager_test_fixture.h>
#include <tests/internal/testing/error_assert.h>

namespace diag_error = orteaf::internal::diagnostics::error;
namespace mps_rt = orteaf::internal::runtime::mps::manager;
namespace mps_wrapper = orteaf::internal::runtime::mps::platform::wrapper;
namespace testing_mps = orteaf::tests::runtime::mps::testing;

using orteaf::tests::ExpectError;

namespace {

mps_wrapper::MpsCommandQueue_t makeQueue(std::uintptr_t value) {
  return reinterpret_cast<mps_wrapper::MpsCommandQueue_t>(value);
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
using ProviderTypes = ::testing::Types<testing_mps::MockBackendOpsProvider,
                                       testing_mps::RealBackendOpsProvider>;
#else
using ProviderTypes = ::testing::Types<testing_mps::MockBackendOpsProvider>;
#endif

} // namespace

TYPED_TEST_SUITE(MpsCommandQueueManagerTypedTest, ProviderTypes);

// =============================================================================
// Configuration Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, GrowthChunkSizeCanBeAdjusted) {
  auto &manager = this->manager();

  // Assert: Default is 1
  EXPECT_EQ(manager.growthChunkSize(), 1u);

  // Act
  manager.setGrowthChunkSize(4);

  // Assert
  EXPECT_EQ(manager.growthChunkSize(), 4u);
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, GrowthChunkSizeRejectsZero) {
  auto &manager = this->manager();

  // Act & Assert
  ExpectError(diag_error::OrteafErrc::InvalidArgument,
              [&] { manager.setGrowthChunkSize(0); });
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           GrowthChunkSizeReflectedInDebugState) {
  if constexpr (!TypeParam::is_mock) {
    GTEST_SKIP() << "Mock-only test";
    return;
  }
  auto &manager = this->manager();

  // Arrange
  manager.setGrowthChunkSize(3);
  const auto device = this->adapter().device();
  this->adapter().expectCreateCommandQueues({makeQueue(0x510)});

  // Act
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Assert
  EXPECT_EQ(manager.growthChunkSize(), 3u);

  // Cleanup
  manager.release(lease);
  this->adapter().expectDestroyCommandQueues({makeQueue(0x510)});
  manager.shutdown();
}

// =============================================================================
// Initialization Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           InitializeCreatesConfiguredNumberOfResources) {
  auto &manager = this->manager();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x1), makeQueue(0x2)});
  const auto device = this->adapter().device();

  // Act
  manager.initialize(device, this->getOps(), 2);

  // Assert
  EXPECT_EQ(manager.capacity(), 2u);

  // Cleanup
  this->adapter().expectDestroyCommandQueuesInOrder(
      {makeQueue(0x1), makeQueue(0x2)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           InitializeRejectsCapacityAboveLimit) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Act & Assert
  ExpectError(diag_error::OrteafErrc::InvalidArgument, [&] {
    manager.initialize(device, this->getOps(),
                       std::numeric_limits<std::size_t>::max());
  });
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, CapacityReflectsPoolSize) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Before init
  EXPECT_EQ(manager.capacity(), 0u);

  // Arrange
  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x301), makeQueue(0x302), makeQueue(0x303)});

  // Act
  manager.initialize(device, this->getOps(), 3);

  // Assert
  EXPECT_EQ(manager.capacity(), 3u);

  // Cleanup
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x301), makeQueue(0x302), makeQueue(0x303)});
  manager.shutdown();
  EXPECT_EQ(manager.capacity(), 0u);
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
           AcquireReturnsDistinctIdsWithinCapacity) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x100), makeQueue(0x200)});
  manager.initialize(device, this->getOps(), 2);

  // Act
  auto id0 = manager.acquire();
  auto id1 = manager.acquire();

  // Assert
  EXPECT_NE(id0.handle(), id1.handle());
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, AcquireGrowsPoolWhenFreelistEmpty) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  manager.setGrowthChunkSize(1);
  this->adapter().expectCreateCommandQueues({makeQueue(0x300)});
  manager.initialize(device, this->getOps(), 1);

  // Act: First acquire uses existing
  auto first = manager.acquire();

  // Arrange: Expect growth
  this->adapter().expectCreateCommandQueues({makeQueue(0x301)});

  // Act: Second acquire triggers growth
  auto second = manager.acquire();

  // Assert
  EXPECT_NE(first.handle(), second.handle());
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           SetGrowthChunkSizeRejectsExcessiveValue) {
  auto &manager = this->manager();

  // Act & Assert: SIZE_MAX exceeds Handle's max index range
  ExpectError(diag_error::OrteafErrc::InvalidArgument, [&] {
    manager.setGrowthChunkSize(std::numeric_limits<std::size_t>::max());
  });
}

// =============================================================================
// Release Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, ReleaseFailsBeforeInitialization) {
  auto &manager = this->manager();

  // Act & Assert
  ExpectError(diag_error::OrteafErrc::InvalidState,
              [&] { (void)manager.acquire(); });
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, ManualReleaseInvalidatesLease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x455)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto lease = manager.acquire();
  const auto original_handle = lease.handle();
  ASSERT_TRUE(static_cast<bool>(lease));

  manager.release(lease);

  // Assert: Lease is invalidated
  EXPECT_FALSE(static_cast<bool>(lease));

  // Assert: With old handle, isAlive returns false (generation mismatch)
  EXPECT_FALSE(manager.isAlive(original_handle));

  // Act: Reacquire gets new generation
  auto reacquired = manager.acquire();
  EXPECT_NE(reacquired.handle(), original_handle);
  reacquired.release();

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0x455)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, ReleaseRejectsNonAcquiredId) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x700)});
  manager.initialize(device, this->getOps(), 1);

  // Act: Move lease and release both
  auto lease = manager.acquire();
  auto moved = std::move(lease);

  // Assert: Release moved-from is benign
  manager.release(lease);
  manager.release(moved);
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, ReleaseRequiresCompletedWork) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x710)});
  manager.initialize(device, this->getOps(), 1);

  // Act & Assert: Release via Lease
  auto lease = manager.acquire();
  lease.release();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           ReleaseMakesHandleStaleAndRecyclesState) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  manager.setGrowthChunkSize(1);
  this->adapter().expectCreateCommandQueues({makeQueue(0x720)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto lease = manager.acquire();
  auto old_handle = lease.handle();
  manager.release(lease);

  this->adapter().expectCreateCommandQueues({});
  auto recycled = manager.acquire();

  // Assert: Handle changed (generation bumped)
  EXPECT_NE(recycled.handle(), old_handle);
}

// =============================================================================
// Capacity Management Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, GrowCapacityAddsAdditionalQueues) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x350)});
  manager.initialize(device, this->getOps(), 1);
  EXPECT_EQ(manager.capacity(), 1u);

  // Act
  this->adapter().expectCreateCommandQueues(
      {makeQueue(0x360), makeQueue(0x370)});
  manager.growCapacity(2);

  // Assert
  EXPECT_EQ(manager.capacity(), 3u);

  // Act: Zero growth is no-op
  manager.growCapacity(0);
  EXPECT_EQ(manager.capacity(), 3u);

  // Act: Verify all can be acquired
  auto id0 = manager.acquire();
  auto id1 = manager.acquire();
  auto id2 = manager.acquire();
  EXPECT_NE(id0.handle(), id1.handle());
  EXPECT_NE(id1.handle(), id2.handle());
  EXPECT_NE(id0.handle(), id2.handle());

  // Cleanup
  id0.release();
  id1.release();
  id2.release();
  this->adapter().expectDestroyCommandQueues(
      {makeQueue(0x350), makeQueue(0x360), makeQueue(0x370)});
  manager.shutdown();
}

// =============================================================================
// Debug State Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, HazardCountersDefaultToZero) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x900)});
  manager.initialize(device, this->getOps(), 1);

  // Act & Assert
  auto lease = manager.acquire();
  lease.release();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           HazardCountersCanBeUpdatedAndResetOnRelease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x910)});
  manager.initialize(device, this->getOps(), 1);

  // Act: Acquire, release, reacquire
  auto lease = manager.acquire();
  lease.release();
  auto recycled = manager.acquire();
  recycled.release();
}

#if ORTEAF_ENABLE_TEST
TYPED_TEST(MpsCommandQueueManagerTypedTest, DebugStateReflectsSetterUpdates) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0x920)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto lease = manager.acquire();
  const auto handle = lease.handle();
  lease.release();

  // Assert: With old handle, isAlive returns false (generation mismatch)
  EXPECT_FALSE(manager.isAlive(handle));
}
#endif

// =============================================================================
// Weak Reference Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           AcquireWeakFromLeaseReturnsValidWeakLease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA00)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto strong_lease = manager.acquire();
  auto weak_lease = manager.acquireWeak(strong_lease);

  // Assert
  EXPECT_TRUE(static_cast<bool>(weak_lease));
  EXPECT_EQ(weak_lease.handle(), strong_lease.handle());

  // Cleanup
  strong_lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA00)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           AcquireWeakFromHandleReturnsValidWeakLease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA10)});
  manager.initialize(device, this->getOps(), 1);

  auto strong_lease = manager.acquire();
  auto handle = strong_lease.handle();

  // Act
  auto weak_lease = manager.acquireWeak(handle);

  // Assert
  EXPECT_TRUE(static_cast<bool>(weak_lease));
  EXPECT_EQ(weak_lease.handle(), handle);

  // Cleanup
  strong_lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA10)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           AcquireWeakFromEmptyLeaseReturnsEmpty) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA20)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  mps_rt::MpsCommandQueueManager::CommandQueueLease empty_lease{};
  auto weak_lease = manager.acquireWeak(empty_lease);

  // Assert
  EXPECT_FALSE(static_cast<bool>(weak_lease));

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA20)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           IsAliveReturnsTrueWhileStrongLeaseHeld) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA30)});
  manager.initialize(device, this->getOps(), 1);

  auto lease = manager.acquire();
  auto handle = lease.handle();

  // Act & Assert
  EXPECT_TRUE(manager.isAlive(handle));

  // Release and check
  lease.release();
  EXPECT_FALSE(manager.isAlive(handle));

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA30)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           TryPromoteSucceedsWhileResourceAlive) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA40)});
  manager.initialize(device, this->getOps(), 1);

  auto strong_lease = manager.acquire();
  auto weak_lease = manager.acquireWeak(strong_lease);
  auto handle = weak_lease.handle();

  // Release strong lease first
  strong_lease.release();

  // Act: Try to promote weak to strong
  auto promoted = manager.tryPromote(handle);

  // Assert: Should succeed because resource still created
  EXPECT_TRUE(static_cast<bool>(promoted));
  EXPECT_EQ(promoted.handle(), handle);

  // Cleanup
  promoted.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA40)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           WeakLeaseDoesNotPreventResourceRelease) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA50)});
  manager.initialize(device, this->getOps(), 1);

  auto strong_lease = manager.acquire();
  auto weak_lease = manager.acquireWeak(strong_lease);
  auto handle = strong_lease.handle();

  // Act: Release strong lease
  strong_lease.release();

  // Assert: Resource is not in use (strong ref released, canTeardown = true)
  const auto &cb = manager.controlBlockForTest(handle.index);
  EXPECT_TRUE(cb.canTeardown());

  // Weak lease still valid but points to released resource
  EXPECT_TRUE(static_cast<bool>(weak_lease));

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA50)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, WeakLeaseCanBeCopied) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA60)});
  manager.initialize(device, this->getOps(), 1);

  auto strong_lease = manager.acquire();
  auto weak1 = manager.acquireWeak(strong_lease);

  // Act: Copy weak lease
  auto weak2 = weak1;

  // Assert
  EXPECT_TRUE(static_cast<bool>(weak1));
  EXPECT_TRUE(static_cast<bool>(weak2));
  EXPECT_EQ(weak1.handle(), weak2.handle());

  // Cleanup
  strong_lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA60)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, MultipleWeakLeasesCanCoexist) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA70)});
  manager.initialize(device, this->getOps(), 1);

  auto strong_lease = manager.acquire();
  auto handle = strong_lease.handle();

  // Act: Create multiple weak leases
  auto weak1 = manager.acquireWeak(strong_lease);
  auto weak2 = manager.acquireWeak(handle);
  auto weak3 = weak1;

  // Assert: All are valid and point to same handle
  EXPECT_TRUE(static_cast<bool>(weak1));
  EXPECT_TRUE(static_cast<bool>(weak2));
  EXPECT_TRUE(static_cast<bool>(weak3));
  EXPECT_EQ(weak1.handle(), handle);
  EXPECT_EQ(weak2.handle(), handle);
  EXPECT_EQ(weak3.handle(), handle);

  // Cleanup
  strong_lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA70)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, TryPromoteFailsForInvalidHandle) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA80)});
  manager.initialize(device, this->getOps(), 1);

  // Act: Try to promote with out-of-range handle
  mps_rt::MpsCommandQueueManager::CommandQueueHandle out_of_range_handle{999,
                                                                         0};
  auto promoted = manager.tryPromote(out_of_range_handle);

  // Assert
  EXPECT_FALSE(static_cast<bool>(promoted));

  // Cleanup
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA80)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, TryPromoteFailsForStaleHandle) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xA90)});
  manager.initialize(device, this->getOps(), 1);

  // Get a lease and release it to make the handle stale
  auto lease = manager.acquire();
  auto stale_handle = lease.handle();
  lease.release();

  // Reacquire to bump generation
  auto new_lease = manager.acquire();
  EXPECT_NE(new_lease.handle(), stale_handle);

  // Act: Try to promote with stale handle (old generation)
  auto promoted = manager.tryPromote(stale_handle);

  // Assert: Should fail because handle is stale (generation mismatch)
  EXPECT_FALSE(static_cast<bool>(promoted));

  // Cleanup
  new_lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xA90)});
  manager.shutdown();
}
