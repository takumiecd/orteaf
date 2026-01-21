#include "orteaf/internal/storage/concepts/storage_concepts.h"
#include "orteaf/internal/storage/cpu/cpu_storage.h"
#include "orteaf/internal/storage/manager/typed_storage_manager.h"
#include "orteaf/internal/storage/registry/storage_types.h"
#include "orteaf/internal/storage/storage_lease.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace storage = orteaf::internal::storage;
namespace concepts = storage::concepts;

// ============================================================
// StorageConcept tests
// ============================================================

TEST(StorageConcepts, CpuStorageSatisfiesConcept) {
  static_assert(concepts::StorageConcept<storage::cpu::CpuStorage>);
}

#if ORTEAF_ENABLE_MPS
TEST(StorageConcepts, MpsStorageSatisfiesConcept) {
  static_assert(concepts::StorageConcept<storage::mps::MpsStorage>);
}
#endif

// ============================================================
// StorageTraits tests
// ============================================================

TEST(StorageTraits, CpuStorageHasName) {
  EXPECT_STREQ(storage::registry::StorageTraits<storage::cpu::CpuStorage>::name,
               "cpu");
}

#if ORTEAF_ENABLE_MPS
TEST(StorageTraits, MpsStorageHasName) {
  EXPECT_STREQ(storage::registry::StorageTraits<storage::mps::MpsStorage>::name,
               "mps");
}
#endif

// ============================================================
// TypedStorageManager type tests
// ============================================================

TEST(TypedStorageManager, CpuManagerTypeExists) {
  using CpuManager = storage::CpuStorageManager;
  static_assert(std::is_class_v<CpuManager>);
}

TEST(TypedStorageManager, CpuLeaseTypeExists) {
  using CpuLease = storage::CpuStorageLease;
  static_assert(std::is_class_v<CpuLease>);
}

// ============================================================
// StorageLease type-erasure tests
// ============================================================

TEST(StorageLease, DefaultConstructedIsInvalid) {
  storage::StorageLease lease;
  EXPECT_FALSE(lease.valid());
}

TEST(StorageLease, ExecutionThrowsWhenInvalid) {
  storage::StorageLease lease;
  EXPECT_THROW((void)lease.execution(), std::system_error);
}

// ============================================================
// RegisteredStorages tests
// ============================================================

TEST(RegisteredStorages, TypeExists) {
  using Registry = storage::RegisteredStorages;
  static_assert(std::is_class_v<Registry>);
}
