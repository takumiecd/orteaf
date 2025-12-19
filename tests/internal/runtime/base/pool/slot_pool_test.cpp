#include "orteaf/internal/runtime/base/pool/slot_pool.h"

#include <gtest/gtest.h>
#include <system_error>

#include "orteaf/internal/base/handle.h"

namespace {

struct SlotTag {};
using SlotHandle = ::orteaf::internal::base::Handle<SlotTag, std::uint32_t, std::uint8_t>;

struct DummyPayload {
  int value{0};
};

struct DummyTraits {
  using Payload = DummyPayload;
  using Handle = SlotHandle;
  struct Request {};
  struct Context {};
  struct Config {
    std::size_t capacity{0};
  };
};

using Pool = ::orteaf::internal::runtime::base::pool::SlotPool<DummyTraits>;

Pool makePool(std::size_t capacity) {
  Pool pool;
  pool.initialize(typename DummyTraits::Config{capacity});
  return pool;
}

TEST(SlotPool, InitializeSetsCapacityAndAvailable) {
  auto pool = makePool(3);
  EXPECT_EQ(pool.capacity(), 3u);
  EXPECT_EQ(pool.available(), 3u);
}

TEST(SlotPool, AcquireReturnsValidSlotRefAndPayloadPtr) {
  auto pool = makePool(2);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto ref = pool.acquire(req, ctx);
  EXPECT_TRUE(ref.valid());
  EXPECT_TRUE(ref.handle.isValid());
  EXPECT_NE(ref.payload_ptr, nullptr);
  EXPECT_EQ(pool.get(ref.handle), ref.payload_ptr);
  EXPECT_EQ(pool.available(), 1u);
}

TEST(SlotPool, TryAcquireReturnsInvalidWhenEmpty) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.tryAcquire(req, ctx);
  auto second = pool.tryAcquire(req, ctx);

  EXPECT_TRUE(first.valid());
  EXPECT_FALSE(second.valid());
  EXPECT_EQ(pool.available(), 0u);
}

TEST(SlotPool, AcquireThrowsWhenEmpty) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  (void)pool.acquire(req, ctx);
  EXPECT_THROW(pool.acquire(req, ctx), std::system_error);
}

TEST(SlotPool, ReleaseReturnsSlotToFreelistAndIncrementsGeneration) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.acquire(req, ctx);
  auto handle = first.handle;

  EXPECT_TRUE(pool.release(handle));

  auto second = pool.acquire(req, ctx);
  EXPECT_EQ(second.handle.index, handle.index);
  EXPECT_EQ(static_cast<std::size_t>(second.handle.generation),
            static_cast<std::size_t>(handle.generation) + 1u);
}

TEST(SlotPool, ReleaseRejectsStaleGeneration) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto first = pool.acquire(req, ctx);
  auto stale = first.handle;

  EXPECT_TRUE(pool.release(stale));
  EXPECT_FALSE(pool.release(stale));
}

TEST(SlotPool, GetReturnsNullForInvalidHandle) {
  auto pool = makePool(1);
  DummyTraits::Request req{};
  DummyTraits::Context ctx{};

  auto ref = pool.acquire(req, ctx);
  auto stale = ref.handle;

  EXPECT_TRUE(pool.release(stale));
  EXPECT_EQ(pool.get(stale), nullptr);
}

} // namespace
