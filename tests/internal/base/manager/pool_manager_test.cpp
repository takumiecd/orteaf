#include "orteaf/internal/base/manager/pool_manager.h"

#include <gtest/gtest.h>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/base/lease/control_block/shared.h"
#include "orteaf/internal/base/pool/slot_pool.h"
#include "orteaf/internal/base/pool/with_control_block_binding.h"
#include "tests/internal/testing/error_assert.h"

namespace {

struct PayloadTag {};
using PayloadHandle =
    ::orteaf::internal::base::Handle<PayloadTag, std::uint32_t, std::uint8_t>;

struct ControlBlockTag {};

struct DummyPayload {
  int value{0};
};

struct DummyPayloadTraits {
  using Payload = DummyPayload;
  using Handle = PayloadHandle;
  struct Request {};
  struct Context {};

  static bool create(Payload &payload, const Request &, const Context &) {
    payload.value = 1;
    return true;
  }

  static void destroy(Payload &payload, const Request &, const Context &) {
    payload.value = 0;
  }
};

using PayloadPool =
    ::orteaf::internal::base::pool::SlotPool<DummyPayloadTraits>;
using ControlBlock =
    ::orteaf::internal::base::SharedControlBlock<PayloadHandle, DummyPayload,
                                                 PayloadPool>;

struct DummyManagerTraits {
  using PayloadPool =
      ::orteaf::internal::base::pool::SlotPool<DummyPayloadTraits>;
  using ControlBlock =
      ::orteaf::internal::base::SharedControlBlock<PayloadHandle, DummyPayload,
                                                   PayloadPool>;
  struct ControlBlockTag {};
  using PayloadHandle = ::PayloadHandle;
  static constexpr const char *Name = "DummyManager";
};

using PoolManager = ::orteaf::internal::base::PoolManager<DummyManagerTraits>;

struct BoundControlBlockTag {};
using BoundControlBlockHandle =
    ::orteaf::internal::base::pool::ControlBlockHandle<BoundControlBlockTag>;
using BoundPayloadPool =
    ::orteaf::internal::base::pool::WithControlBlockBinding<
        PayloadPool, BoundControlBlockHandle>;
using BoundControlBlock =
    ::orteaf::internal::base::SharedControlBlock<PayloadHandle, DummyPayload,
                                                 BoundPayloadPool>;

struct BoundManagerTraits {
  using PayloadPool = BoundPayloadPool;
  using ControlBlock = BoundControlBlock;
  using ControlBlockTag = BoundControlBlockTag;
  using PayloadHandle = ::PayloadHandle;
  static constexpr const char *Name = "BoundManager";
};

using BoundPoolManager =
    ::orteaf::internal::base::PoolManager<BoundManagerTraits>;

// Builder type aliases
using Builder = PoolManager::Builder<DummyPayloadTraits::Request,
                                     DummyPayloadTraits::Context>;
using BoundBuilder = BoundPoolManager::Builder<DummyPayloadTraits::Request,
                                               DummyPayloadTraits::Context>;

Builder makeBaseBuilder() {
  return Builder{}
      .withControlBlockCapacity(2)
      .withControlBlockBlockSize(2)
      .withControlBlockGrowthChunkSize(1)
      .withPayloadGrowthChunkSize(1)
      .withPayloadCapacity(2)
      .withPayloadBlockSize(2);
}

TEST(PoolManager, InitiallyNotConfigured) {
  PoolManager manager;
  EXPECT_FALSE(manager.isConfigured());
}

TEST(PoolManager, EnsureConfiguredThrowsWhenNotConfigured) {
  PoolManager manager;
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "has not been configured"},
      [&manager] { manager.ensureConfigured(); });
}

TEST(PoolManager, ConfigureRejectsZeroControlBlockBlockSize) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withControlBlockBlockSize(0)
                     .withRequest(req)
                     .withContext(ctx);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "control block size must be > 0"},
      [&manager, &builder] { builder.configure(manager); });
}

TEST(PoolManager, ConfigureRejectsZeroControlBlockGrowthChunkSize) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withControlBlockGrowthChunkSize(0)
                     .withRequest(req)
                     .withContext(ctx);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "control block growth chunk size must be > 0"},
      [&manager, &builder] { builder.configure(manager); });
}

TEST(PoolManager, ConfigureRejectsZeroPayloadGrowthChunkSize) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withPayloadGrowthChunkSize(0)
                     .withRequest(req)
                     .withContext(ctx);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "payload growth chunk size must be > 0"},
      [&manager, &builder] { builder.configure(manager); });
}

TEST(PoolManager, ConfigureRejectsZeroPayloadBlockSize) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder =
      makeBaseBuilder().withPayloadBlockSize(0).withRequest(req).withContext(
          ctx);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "payload block size must be > 0"},
      [&manager, &builder] { builder.configure(manager); });
}

TEST(PoolManager, ConfigureMarksManagerConfigured) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  EXPECT_TRUE(manager.isConfigured());
}

TEST(PoolManager, ConfigureWithZeroCapacityMarksConfiguredAndAllowsGrowth) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withPayloadCapacity(0)
                     .withControlBlockCapacity(0)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  EXPECT_TRUE(manager.isConfigured());

  auto handle = manager.acquirePayloadOrGrowAndCreate(req, ctx);
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.isAlive(handle));

  auto lease = manager.acquireStrongLease(handle);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.operator->(), nullptr);
  EXPECT_EQ(lease->value, 1);

  lease.release();
  manager.shutdown(req, ctx);
  EXPECT_FALSE(manager.isConfigured());
}

TEST(PoolManager, SetControlBlockBlockSizeRejectsZero) {
  PoolManager manager;
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "control block size must be > 0"},
      [&manager] { manager.setControlBlockBlockSize(0); });
}

TEST(PoolManager, SetControlBlockBlockSizeAcceptsNonZero) {
  PoolManager manager;
  EXPECT_NO_THROW(manager.setControlBlockBlockSize(2));
}

TEST(PoolManager, SetControlBlockBlockSizeRejectsWhenLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "shutdown aborted due to active leases"},
      [&manager] { manager.setControlBlockBlockSize(3); });
}

TEST(PoolManager, SetControlBlockBlockSizeAcceptsAfterLeaseReleased) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  {
    auto lease = manager.acquireStrongLease(handle);
    EXPECT_TRUE(lease);
  }

  EXPECT_NO_THROW(manager.setControlBlockBlockSize(3));
}

TEST(PoolManager, SetPayloadBlockSizeRejectsZero) {
  PoolManager manager;
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "payload block size must be > 0"},
      [&manager] { manager.setPayloadBlockSize(0); });
}

TEST(PoolManager, SetPayloadBlockSizeAcceptsNonZero) {
  PoolManager manager;
  EXPECT_NO_THROW(manager.setPayloadBlockSize(2));
}

TEST(PoolManager, SetPayloadBlockSizeRejectsWhenLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "teardown aborted due to active strong references"},
      [&manager] { manager.setPayloadBlockSize(3); });
}

TEST(PoolManager, SetPayloadBlockSizeAcceptsAfterLeaseReleased) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  {
    auto lease = manager.acquireStrongLease(handle);
    EXPECT_TRUE(lease);
  }

  EXPECT_NO_THROW(manager.setPayloadBlockSize(3));
}

TEST(PoolManager, ConfigureRejectsControlBlockSizeChangeWhenLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireWeakLease(handle);
  EXPECT_TRUE(lease);

  auto updated = makeBaseBuilder()
                     .withControlBlockBlockSize(3)
                     .withRequest(req)
                     .withContext(ctx);
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "shutdown aborted due to active leases"},
      [&manager, &updated] { updated.configure(manager); });
}

TEST(PoolManager, ConfigureAcceptsControlBlockSizeChangeAfterLeaseReleased) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  {
    auto lease = manager.acquireWeakLease(handle);
    EXPECT_TRUE(lease);
  }

  auto updated = makeBaseBuilder()
                     .withControlBlockBlockSize(3)
                     .withRequest(req)
                     .withContext(ctx);
  EXPECT_NO_THROW(updated.configure(manager));
}

TEST(PoolManager, ConfigureRejectsPayloadBlockSizeChangeWhenStrongLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);

  auto updated =
      makeBaseBuilder().withPayloadBlockSize(3).withRequest(req).withContext(
          ctx);
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "teardown aborted due to active strong references"},
      [&manager, &updated] { updated.configure(manager); });
}

TEST(PoolManager, ConfigureAcceptsPayloadBlockSizeChangeAfterLeaseReleased) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  {
    auto lease = manager.acquireStrongLease(handle);
    EXPECT_TRUE(lease);
  }

  auto updated =
      makeBaseBuilder().withPayloadBlockSize(3).withRequest(req).withContext(
          ctx);
  EXPECT_NO_THROW(updated.configure(manager));
}

TEST(PoolManager, ShutdownRejectsWhenStrongLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "teardown aborted due to active strong references"},
      [&manager, &req, &ctx] { manager.shutdown(req, ctx); });
}

TEST(PoolManager, ShutdownRejectsWhenWeakLeaseActive) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto strong = manager.acquireStrongLease(handle);
  EXPECT_TRUE(strong);
  auto weak = manager.acquireWeakLease(handle);
  EXPECT_TRUE(weak);
  strong.release();

  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "shutdown aborted due to active leases"},
      [&manager, &req, &ctx] { manager.shutdown(req, ctx); });
}

TEST(PoolManager, ShutdownAcceptsAfterLeasesReleased) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  {
    auto strong = manager.acquireStrongLease(handle);
    EXPECT_TRUE(strong);
    auto weak = manager.acquireWeakLease(handle);
    EXPECT_TRUE(weak);
  }

  EXPECT_NO_THROW(manager.shutdown(req, ctx));
  EXPECT_FALSE(manager.isConfigured());
}

TEST(PoolManager, ControlBlockGrowthChunkSizeDefaultsToOne) {
  PoolManager manager;
  EXPECT_EQ(manager.controlBlockGrowthChunkSize(), 1u);
}

TEST(PoolManager, SetControlBlockGrowthChunkSizeRejectsZero) {
  PoolManager manager;
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "control block growth chunk size must be > 0"},
      [&manager] { manager.setControlBlockGrowthChunkSize(0); });
}

TEST(PoolManager, SetControlBlockGrowthChunkSizeUpdatesValue) {
  PoolManager manager;
  manager.setControlBlockGrowthChunkSize(3);
  EXPECT_EQ(manager.controlBlockGrowthChunkSize(), 3u);
}

TEST(PoolManager, PayloadGrowthChunkSizeDefaultsToOne) {
  PoolManager manager;
  EXPECT_EQ(manager.payloadGrowthChunkSize(), 1u);
}

TEST(PoolManager, SetPayloadGrowthChunkSizeRejectsZero) {
  PoolManager manager;
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "payload growth chunk size must be > 0"},
      [&manager] { manager.setPayloadGrowthChunkSize(0); });
}

TEST(PoolManager, SetPayloadGrowthChunkSizeUpdatesValue) {
  PoolManager manager;
  manager.setPayloadGrowthChunkSize(4);
  EXPECT_EQ(manager.payloadGrowthChunkSize(), 4u);
}

TEST(PoolManager, IsAliveReturnsFalseWhenNotConfigured) {
  PoolManager manager;
  EXPECT_FALSE(manager.isAlive(PayloadHandle::invalid()));
}

TEST(PoolManager, IsAliveReturnsFalseForInvalidHandle) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  EXPECT_FALSE(manager.isAlive(PayloadHandle::invalid()));
}

TEST(PoolManager, IsAliveReturnsFalseForUncreatedPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_FALSE(manager.isAlive(handle));
}

TEST(PoolManager, IsAliveReturnsTrueForCreatedPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));
  EXPECT_TRUE(manager.isAlive(handle));
}

TEST(PoolManager, EmplacePayloadReturnsTrueAndCreatesPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));
  EXPECT_TRUE(manager.isAlive(handle));
}

TEST(PoolManager, EmplacePayloadReturnsFalseForInvalidHandle) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  EXPECT_FALSE(manager.emplacePayload(PayloadHandle::invalid(), req, ctx));
}

TEST(PoolManager, EmplacePayloadReturnsFalseWhenAlreadyCreated) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));
  EXPECT_FALSE(manager.emplacePayload(handle, req, ctx));
}

TEST(PoolManager, CreateAllPayloadsCreatesAllSlots) {
  PoolManager manager;
  constexpr std::size_t kPayloadCapacity = 3;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withPayloadCapacity(kPayloadCapacity)
                     .withPayloadBlockSize(kPayloadCapacity)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  EXPECT_TRUE(manager.createAllPayloads(req, ctx));

  for (std::size_t i = 0; i < kPayloadCapacity; ++i) {
    auto handle = PayloadHandle{static_cast<PayloadHandle::index_type>(i), 0};
    EXPECT_TRUE(manager.isAlive(handle));
  }
}

TEST(PoolManager, ReserveUncreatedPayloadOrGrowReturnsUncreatedHandle) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_FALSE(manager.isAlive(handle));
}

TEST(PoolManager, ReserveUncreatedPayloadOrGrowGrowsWhenExhausted) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withPayloadCapacity(1)
                     .withPayloadGrowthChunkSize(2)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto first = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(first.isValid());
  auto second = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(second.isValid());
  EXPECT_NE(second.index, first.index);
  EXPECT_FALSE(manager.isAlive(second));
}

TEST(PoolManager, AcquirePayloadOrGrowAndCreateCreatesPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder()
                     .withPayloadCapacity(0)
                     .withPayloadBlockSize(1)
                     .withPayloadGrowthChunkSize(1)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto handle = manager.acquirePayloadOrGrowAndCreate(req, ctx);
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.isAlive(handle));
}

TEST(PoolManager, AcquireStrongLeaseRejectsInvalidHandle) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "handle is invalid"},
      [&manager] { manager.acquireStrongLease(PayloadHandle::invalid()); });
}

TEST(PoolManager, AcquireWeakLeaseRejectsInvalidHandle) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      {"DummyManager", "handle is invalid"},
      [&manager] { manager.acquireWeakLease(PayloadHandle::invalid()); });
}

TEST(PoolManager, AcquireStrongLeaseRejectsUncreatedPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "payload is unavailable"},
      [&manager, handle] { manager.acquireStrongLease(handle); });
}

TEST(PoolManager, AcquireWeakLeaseRejectsUncreatedPayload) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  ::orteaf::tests::ExpectErrorMessage(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      {"DummyManager", "payload is unavailable"},
      [&manager, handle] { manager.acquireWeakLease(handle); });
}

TEST(PoolManager, AcquireStrongLeaseReturnsValidLease) {
  PoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = makeBaseBuilder().withRequest(req).withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);
  EXPECT_EQ(lease.strongCount(), 1u);
  EXPECT_NE(lease.operator->(), nullptr);
  EXPECT_EQ(lease->value, 1);
}

TEST(PoolManager, AcquireWeakLeaseReturnsValidLease) {
  BoundPoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = BoundBuilder{}
                     .withControlBlockCapacity(2)
                     .withControlBlockBlockSize(2)
                     .withControlBlockGrowthChunkSize(1)
                     .withPayloadGrowthChunkSize(1)
                     .withPayloadCapacity(2)
                     .withPayloadBlockSize(2)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  // First get a StrongLease - this creates the control block binding
  auto existingStrong = manager.acquireStrongLease(handle);
  EXPECT_TRUE(existingStrong);

  // Now get a WeakLease - shares the same control block
  auto lease = manager.acquireWeakLease(handle);
  EXPECT_TRUE(lease);
  // weak count may be higher due to internal implementation
  EXPECT_EQ(lease.weakCount(), 1u);

  // lock() will work because strong count > 0
  auto strong = lease.lock();
  ASSERT_TRUE(strong);
  EXPECT_NE(strong.operator->(), nullptr);
  EXPECT_EQ(strong->value, 1);

  // Release all
  strong.release();
  existingStrong.release();
}

TEST(PoolManager, AcquireStrongLeaseRebindsControlBlockAfterRelease) {
  BoundPoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = BoundBuilder{}
                     .withControlBlockCapacity(1)
                     .withControlBlockBlockSize(1)
                     .withControlBlockGrowthChunkSize(1)
                     .withPayloadGrowthChunkSize(1)
                     .withPayloadCapacity(1)
                     .withPayloadBlockSize(1)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto lease = manager.acquireStrongLease(handle);
  EXPECT_TRUE(lease);
  auto bound = manager.boundControlBlockForTest(handle);
  EXPECT_TRUE(bound.isValid());
  EXPECT_EQ(bound, lease.handle());

  lease.release();
  auto reacquired_handle = manager.acquirePayloadOrGrowAndCreate(req, ctx);
  EXPECT_TRUE(reacquired_handle.isValid());
  EXPECT_EQ(reacquired_handle.index, handle.index);
  auto bound_after = manager.boundControlBlockForTest(reacquired_handle);
  EXPECT_FALSE(bound_after.isValid());

  auto reused = manager.acquireStrongLease(reacquired_handle);
  EXPECT_TRUE(reused);
  auto rebound = manager.boundControlBlockForTest(reacquired_handle);
  EXPECT_TRUE(rebound.isValid());
  EXPECT_EQ(rebound, reused.handle());
  EXPECT_EQ(reused.strongCount(), 1u);
}

TEST(PoolManager, AcquireStrongLeaseTwiceReusesSameControlBlockAndHandle) {
  BoundPoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = BoundBuilder{}
                     .withControlBlockCapacity(1)
                     .withControlBlockBlockSize(1)
                     .withControlBlockGrowthChunkSize(1)
                     .withPayloadGrowthChunkSize(1)
                     .withPayloadCapacity(1)
                     .withPayloadBlockSize(1)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto first = manager.acquireStrongLease(handle);
  auto second = manager.acquireStrongLease(handle);

  EXPECT_TRUE(first);
  EXPECT_TRUE(second);
  EXPECT_EQ(first.handle(), second.handle());
  EXPECT_EQ(first.payloadHandle(), handle);
  EXPECT_EQ(second.payloadHandle(), handle);
}

TEST(PoolManager, AcquireWeakLeaseTwiceReusesSameControlBlockAndHandle) {
  BoundPoolManager manager;
  DummyPayloadTraits::Request req{};
  DummyPayloadTraits::Context ctx{};
  auto builder = BoundBuilder{}
                     .withControlBlockCapacity(1)
                     .withControlBlockBlockSize(1)
                     .withControlBlockGrowthChunkSize(1)
                     .withPayloadGrowthChunkSize(1)
                     .withPayloadCapacity(1)
                     .withPayloadBlockSize(1)
                     .withRequest(req)
                     .withContext(ctx);

  builder.configure(manager);
  auto handle = manager.reserveUncreatedPayloadOrGrow();
  EXPECT_TRUE(handle.isValid());
  EXPECT_TRUE(manager.emplacePayload(handle, req, ctx));

  auto first = manager.acquireWeakLease(handle);
  auto second = manager.acquireWeakLease(handle);

  EXPECT_TRUE(first);
  EXPECT_TRUE(second);
  EXPECT_EQ(first.handle(), second.handle());

  // lock() requires active strong references, so get one first
  auto existingStrong = manager.acquireStrongLease(handle);
  EXPECT_TRUE(existingStrong);

  auto firstStrong = first.lock();
  auto secondStrong = second.lock();
  ASSERT_TRUE(firstStrong);
  ASSERT_TRUE(secondStrong);
  EXPECT_EQ(firstStrong.payloadHandle(), handle);
  EXPECT_EQ(secondStrong.payloadHandle(), handle);

  // Cleanup
  firstStrong.release();
  secondStrong.release();
  existingStrong.release();
}

} // namespace
