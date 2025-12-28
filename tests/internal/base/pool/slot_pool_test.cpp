#include "orteaf/internal/base/pool/slot_pool.h"

#include <gtest/gtest.h>
#include <system_error>
#include <vector>

#include "orteaf/internal/base/handle.h"

namespace {

struct SlotTag {};
using SlotHandle =
    ::orteaf::internal::base::Handle<SlotTag, std::uint32_t, std::uint8_t>;

struct DummyPayload {
  int value{0};
};

struct DummyTraits {
  using Payload = DummyPayload;
  using Handle = SlotHandle;
  struct Request {};
  struct Context {};

  static bool create(Payload &payload, const Request &, const Context &) {
    payload.value = 42;
    return true;
  }

  static void destroy(Payload &payload, const Request &, const Context &) {
    payload.value = 0;
  }
};

struct DestroyOnReleaseTraits {
  using Payload = DummyPayload;
  using Handle = SlotHandle;
  struct Request {};
  struct Context {};
  static constexpr bool destroy_on_release = true;

  static bool create(Payload &payload, const Request &, const Context &) {
    payload.value = 9;
    return true;
  }

  static void destroy(Payload &payload, const Request &, const Context &) {
    payload.value = -1;
  }
};

using Pool = ::orteaf::internal::base::pool::SlotPool<DummyTraits>;
using DestroyOnReleasePool =
    ::orteaf::internal::base::pool::SlotPool<DestroyOnReleaseTraits>;

Pool makePool(std::size_t capacity) {
  Pool pool;
  pool.setBlockSize(capacity);
  pool.resize(capacity);
  return pool;
}

DestroyOnReleasePool makeDestroyOnReleasePool(std::size_t capacity) {
  DestroyOnReleasePool pool;
  pool.setBlockSize(capacity);
  pool.resize(capacity);
  return pool;
}

TEST(SlotPool, InitializeSetsSizeAndAvailable) {
  auto pool = makePool(3);
  EXPECT_EQ(pool.size(), 3u);
  EXPECT_EQ(pool.available(), 3u);
}

TEST(SlotPool, ReserveReturnsValidHandleAndPayloadPtr) {
  auto pool = makePool(2);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto handle = pool.reserveUncreated();
  EXPECT_TRUE(handle.isValid());
  EXPECT_NE(pool.get(handle), nullptr);
  EXPECT_EQ(pool.available(), 1u);
}

TEST(SlotPool, TryReserveReturnsInvalidWhenEmpty) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.tryReserveUncreated();
  auto second = pool.tryReserveUncreated();

  EXPECT_TRUE(first.isValid());
  EXPECT_FALSE(second.isValid());
  EXPECT_EQ(pool.available(), 0u);
}

TEST(SlotPool, ReserveThrowsWhenEmpty) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  (void)pool.reserveUncreated();
  EXPECT_THROW(pool.reserveUncreated(), std::system_error);
}

TEST(SlotPool, TryAcquireReturnsInvalidWhenNoCreated) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto handle = pool.tryAcquireCreated();
  EXPECT_FALSE(handle.isValid());
}

TEST(SlotPool, AcquireThrowsWhenNoCreated) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  EXPECT_THROW(pool.acquireCreated(), std::system_error);
}

TEST(SlotPool, AcquireReturnsCreatedSlotAfterRelease) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto reserved = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(reserved, req, ctx));
  EXPECT_TRUE(pool.release(reserved));

  auto handle = pool.acquireCreated();
  EXPECT_TRUE(handle.isValid());
}

TEST(SlotPool, ReleaseReturnsSlotToFreelistAndIncrementsGeneration) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(first, req, ctx));

  EXPECT_TRUE(pool.release(first));

  auto second = pool.acquireCreated();
  EXPECT_EQ(second.index, first.index);
  EXPECT_EQ(static_cast<std::size_t>(second.generation),
            static_cast<std::size_t>(first.generation) + 1u);
}

TEST(SlotPool, ReleaseRejectsStaleGeneration) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(first, req, ctx));

  EXPECT_TRUE(pool.release(first));
  EXPECT_FALSE(pool.release(first));
}

TEST(SlotPool, EmplaceUsesTraitsCreateAndSetsCreated) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(slot, req, ctx));
  EXPECT_TRUE(pool.isCreated(slot));
  EXPECT_EQ(pool.get(slot)->value, 42);
}

TEST(SlotPool, EmplaceLambdaOverridesTraitsCreate) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_TRUE(
      pool.emplace(slot, req, ctx,
                   [](DummyPayload &payload, const DummyTraits::Request &,
                      const DummyTraits::Context &) {
                     payload.value = 7;
                     return true;
                   }));
  EXPECT_TRUE(pool.isCreated(slot));
  EXPECT_EQ(pool.get(slot)->value, 7);
}

TEST(SlotPool, DestroyUsesTraitsDestroyAndClearsCreated) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(slot, req, ctx));
  EXPECT_TRUE(pool.destroy(slot, req, ctx));
  EXPECT_FALSE(pool.isCreated(slot));
  EXPECT_EQ(pool.get(slot)->value, 0);
}

TEST(SlotPool, DestroyLambdaOverridesTraitsDestroy) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(slot, req, ctx));
  EXPECT_TRUE(
      pool.destroy(slot, req, ctx,
                   [](DummyPayload &payload, const DummyTraits::Request &,
                      const DummyTraits::Context &) {
                     payload.value = -1;
                     return true;
                   }));
  EXPECT_FALSE(pool.isCreated(slot));
  EXPECT_EQ(pool.get(slot)->value, -1);
}

TEST(SlotPool, GetReturnsNullForInvalidHandle) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto handle = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(handle, req, ctx));

  EXPECT_TRUE(pool.release(handle));
  EXPECT_EQ(pool.get(handle), nullptr);
}

TEST(SlotPool, ReleaseDestroysWhenConfigured) {
  auto pool = makeDestroyOnReleasePool(1);
  DestroyOnReleaseTraits::Request req{};
  DestroyOnReleaseTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(slot, req, ctx));
  EXPECT_TRUE(pool.release(slot, req, ctx));
  EXPECT_FALSE(pool.isCreated(slot));
  auto re_reserved = pool.reserveUncreated();
  EXPECT_TRUE(re_reserved.isValid());
  EXPECT_EQ(pool.get(re_reserved)->value, -1);
}

TEST(SlotPool, ReleaseFailsWhenNotCreatedInDestroyMode) {
  auto pool = makeDestroyOnReleasePool(1);
  DestroyOnReleaseTraits::Request req{};
  DestroyOnReleaseTraits::Context ctx{};

  auto slot = pool.reserveUncreated();
  EXPECT_FALSE(pool.release(slot, req, ctx));
}

TEST(SlotPool, GrowAddsUncreatedSlots) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  pool.setBlockSize(3);
  pool.resize(3);

  EXPECT_EQ(pool.size(), 3u);
  EXPECT_EQ(pool.available(), 3u);

  auto handle = pool.reserveUncreated();
  EXPECT_TRUE(handle.isValid());
  EXPECT_FALSE(pool.isCreated(handle));
}

TEST(SlotPool, GrowAndCreateCreatesNewSlots) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto reserved = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(reserved, req, ctx));
  EXPECT_TRUE(pool.release(reserved));

  pool.setBlockSize(3);
  const std::size_t old_capacity = pool.resize(3);
  EXPECT_TRUE(pool.createRange(old_capacity, pool.size(), req, ctx));
  EXPECT_FALSE(pool.tryReserveUncreated().isValid());

  auto handle = pool.acquireCreated();
  EXPECT_TRUE(handle.isValid());
}

TEST(SlotPool, ForEachCreatedVisitsOnlyCreatedSlots) {
  auto pool = makePool(3);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(first, req, ctx));

  auto second = pool.reserveUncreated();
  EXPECT_TRUE(pool.emplace(second, req, ctx));
  EXPECT_TRUE(pool.release(second));

  std::vector<std::size_t> indices{};
  std::vector<int> values{};
  pool.forEachCreated([&](std::size_t idx, const DummyPayload &payload) {
    indices.push_back(idx);
    values.push_back(payload.value);
  });

  ASSERT_EQ(indices.size(), 2u);
  EXPECT_EQ(indices[0], static_cast<std::size_t>(first.index));
  EXPECT_EQ(values[0], 42);
}

TEST(SlotPool, InitializeAndCreateCreatesAllSlots) {
  Pool pool;
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  pool.setBlockSize(2);
  pool.resize(2);
  EXPECT_TRUE(pool.createAll(req, ctx));
  EXPECT_EQ(pool.size(), 2u);
  EXPECT_EQ(pool.available(), 2u);
  EXPECT_FALSE(pool.tryReserveUncreated().isValid());
  EXPECT_TRUE(pool.tryAcquireCreated().isValid());
}

TEST(SlotPool, ReserveDoesNotChangeSize) {
  Pool pool;
  pool.reserve(4);
  EXPECT_EQ(pool.size(), 0u);
  EXPECT_GE(pool.capacity(), 4u);
}

TEST(SlotPool, ResizeGrowsSlots) {
  Pool pool;
  const std::size_t old_size = pool.resize(3);
  EXPECT_EQ(old_size, 0u);
  EXPECT_EQ(pool.size(), 3u);
  EXPECT_EQ(pool.available(), 3u);
}

} // namespace
