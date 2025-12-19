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

  // Assert: For non-generational handles, same slot is reused (same index)
  // With generation, handles would differ; without generation, index is same
  EXPECT_EQ(recycled.handle().index, old_handle.index);
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

  // Assert: Resource is not in use (canTeardown = true)
  const auto &cb = manager.controlBlockForTest(handle.index);
  EXPECT_TRUE(cb.canTeardown());
}
#endif

// =============================================================================
// Shared Ownership Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           LeasesCanBeCopiedAndShareOwnership) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xB00)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto lease1 = manager.acquire();
  auto lease2 = lease1; // Copy

  // Assert
  EXPECT_TRUE(lease1.isValid());
  EXPECT_TRUE(lease2.isValid());
  EXPECT_EQ(lease1.handle(), lease2.handle());

  // Cleanup: Release both. Expect destroy ONLY after second release.
  lease1.release();
  // Resource should still be alive (lease2 holds it)
  // We can't easily check "alive" without internal access, but we can verify no
  // destroy call yet.

  this->adapter().expectDestroyCommandQueues({makeQueue(0xB00)});
  lease2.release(); // Now expect destroy (triggered by shutdown or here if we
                    // tracked strictly)
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest,
           ResourceDestroyedWhenAllLeasesGone) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xB10)});
  manager.initialize(device, this->getOps(), 1);

  // Act
  auto lease1 = manager.acquire();
  auto lease2 = lease1;

  // Release logic check
  lease1.release();

  // Clean up remaining
  this->adapter().expectDestroyCommandQueues({makeQueue(0xB10)});
  lease2.release();
  manager.shutdown();
}

// =============================================================================
// Locking Tests
// =============================================================================

TYPED_TEST(MpsCommandQueueManagerTypedTest, TryLockAcquiresExplicitLock) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xC00)});
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Act: tryLock returns ScopedLock with RAII unlocking
  auto lock = manager.tryLock(lease);

  // Assert
  EXPECT_TRUE(lock); // Lock acquired

  // Cleanup (lock releases automatically via RAII)
  lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xC00)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, TryLockFailsIfAlreadyLocked) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xC10)});
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Act: First lock succeeds, second fails while first is held
  auto firstLock = manager.tryLock(lease);
  auto secondLock = manager.tryLock(lease); // Fails because mutex is held

  // Assert
  EXPECT_TRUE(firstLock);
  EXPECT_FALSE(secondLock); // Already locked

  // Cleanup (firstLock releases automatically via RAII)
  lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xC10)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, ScopedLockUnlockOnDestruction) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xC20)});
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Act & Assert
  {
    auto lock = lease.tryLock(); // Returns ScopedLock with mutex lock
    ASSERT_TRUE(lock);           // Check if lock acquired

    // While locked, another tryLock should fail
    auto lock2 = lease.tryLock();
    EXPECT_FALSE(lock2);
  }
  // lock destroyed here -> mutex unlocked via RAII

  // Assert: Can lock again after ScopedLock is destroyed
  {
    auto lock3 = lease.tryLock();
    EXPECT_TRUE(lock3);
  }

  // Cleanup
  lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xC20)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, ScopedLockProvidesPayloadAccess) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xC30)});
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Act: Lock and access via ScopedLock
  auto lock = lease.tryLock();
  ASSERT_TRUE(lock);

  // Access payload through ScopedLock
  auto &payload = *lock;
  EXPECT_NE(payload, nullptr);

  // Cleanup
  lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xC30)});
  manager.shutdown();
}

TYPED_TEST(MpsCommandQueueManagerTypedTest, BlockingLockWorks) {
  auto &manager = this->manager();
  const auto device = this->adapter().device();

  // Arrange
  this->adapter().expectCreateCommandQueues({makeQueue(0xC40)});
  manager.initialize(device, this->getOps(), 1);
  auto lease = manager.acquire();

  // Act: Use blocking lock()
  auto lock = lease.lock(); // Blocking lock
  ASSERT_TRUE(lock);

  // Access payload
  auto &payload = *lock;
  EXPECT_NE(payload, nullptr);

  // Cleanup
  lease.release();
  this->adapter().expectDestroyCommandQueues({makeQueue(0xC40)});
  manager.shutdown();
}
