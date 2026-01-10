#include "orteaf/internal/storage/storage.h"
#include "orteaf/internal/storage/storage_types.h"

#include <gtest/gtest.h>

namespace storage = orteaf::internal::storage;

// ============================================================
// StorageType concept tests
// ============================================================

TEST(StorageTypes, CpuStorageSatisfiesConcept) {
  static_assert(storage::StorageType<storage::CpuStorage>);
  static_assert(storage::StorageType<storage::CpuStorage &>);
  static_assert(storage::StorageType<const storage::CpuStorage &>);
  static_assert(storage::StorageType<storage::CpuStorage &&>);
}

TEST(StorageTypes, MonostateDoeNotSatisfyConcept) {
  static_assert(!storage::StorageType<std::monostate>);
}

// ============================================================
// Storage class tests
// ============================================================

TEST(Storage, DefaultConstructedIsInvalid) {
  storage::Storage s;
  EXPECT_FALSE(s.valid());
}

TEST(Storage, EraseFromCpuStorage) {
  storage::CpuStorage cpu_storage;
  storage::Storage s = storage::Storage::erase(std::move(cpu_storage));
  EXPECT_TRUE(s.valid());
}

TEST(Storage, TryAsCpuStorage) {
  storage::CpuStorage cpu_storage;
  storage::Storage s = storage::Storage::erase(std::move(cpu_storage));

  auto *ptr = s.tryAs<storage::CpuStorage>();
  EXPECT_NE(ptr, nullptr);
}

TEST(Storage, TryAsWrongTypeReturnsNull) {
  storage::CpuStorage cpu_storage;
  storage::Storage s = storage::Storage::erase(std::move(cpu_storage));

#if ORTEAF_ENABLE_MPS
  auto *ptr = s.tryAs<storage::MpsStorage>();
  EXPECT_EQ(ptr, nullptr);
#endif
}

TEST(Storage, VisitPattern) {
  storage::CpuStorage cpu_storage;
  storage::Storage s = storage::Storage::erase(std::move(cpu_storage));

  bool visited_cpu = false;
  s.visit([&](auto &storage) {
    using T = std::decay_t<decltype(storage)>;
    if constexpr (std::is_same_v<T, storage::CpuStorage>) {
      visited_cpu = true;
    }
  });

  EXPECT_TRUE(visited_cpu);
}

TEST(Storage, VisitPatternOnInvalid) {
  storage::Storage s;

  bool visited_monostate = false;
  s.visit([&](auto &storage) {
    using T = std::decay_t<decltype(storage)>;
    if constexpr (std::is_same_v<T, std::monostate>) {
      visited_monostate = true;
    }
  });

  EXPECT_TRUE(visited_monostate);
}
