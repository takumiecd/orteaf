#include "orteaf/internal/kernel/registry/kernel_registry.h"

#include <gtest/gtest.h>

#include <type_traits>

#include "orteaf/internal/kernel/core/kernel_key.h"
#include "orteaf/internal/kernel/kernel_metadata.h"
#include "orteaf/internal/kernel/registry/kernel_registry_config.h"

namespace registry = orteaf::internal::kernel::registry;
namespace kernel = orteaf::internal::kernel;

namespace {

// Helper to create test kernel key
kernel::KernelKey makeKey(int id) {
  return static_cast<kernel::KernelKey>(static_cast<std::uint64_t>(id) << 12);
}

// Helper to create test metadata
kernel::KernelMetadataLease makeMetadata(const char *lib, const char *func) {
  (void)lib;
  (void)func;
  return kernel::KernelMetadataLease{};
}

static_assert(!std::is_move_constructible_v<registry::KernelRegistry>);
static_assert(!std::is_move_assignable_v<registry::KernelRegistry>);

// ============================================================
// Basic construction tests
// ============================================================

TEST(KernelRegistryTest, DefaultConstruction) {
  registry::KernelRegistry reg;

  EXPECT_EQ(reg.cacheSize(), 0u);
  EXPECT_EQ(reg.mainMemorySize(), 0u);
  EXPECT_EQ(reg.secondaryStorageSize(), 0u);
  EXPECT_EQ(reg.config().cache_capacity, 8u);
  EXPECT_EQ(reg.config().main_memory_capacity, 64u);
}

TEST(KernelRegistryTest, CustomConfiguration) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 4;
  config.main_memory_capacity = 16;

  registry::KernelRegistry reg(config);

  EXPECT_EQ(reg.config().cache_capacity, 4u);
  EXPECT_EQ(reg.config().main_memory_capacity, 16u);
}

#if ORTEAF_ENABLE_TESTING
TEST(KernelRegistryTest, CacheNodePointerStableAcrossRehash) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 128;
  config.main_memory_capacity = 128;

  registry::KernelRegistry reg(config);
  auto key = makeKey(1);
  reg.registerKernel(key, makeMetadata("lib1", "func1"));
  ASSERT_NE(reg.lookup(key), nullptr);

  auto &nodes = reg.cacheNodesForTest();
  auto node_it = nodes.find(key);
  ASSERT_NE(node_it, nodes.end());

  auto *node_ptr = node_it->second.get();
  ASSERT_NE(node_ptr, nullptr);

  auto before_buckets = nodes.bucket_count();
  nodes.rehash(before_buckets * 2 + 1);
  EXPECT_GT(nodes.bucket_count(), before_buckets);

  auto node_it_after = nodes.find(key);
  ASSERT_NE(node_it_after, nodes.end());
  EXPECT_EQ(node_it_after->second.get(), node_ptr);
  EXPECT_NE(reg.lookup(key), nullptr);
}

TEST(KernelRegistryTest, MainMemoryNodePointerStableAcrossRehash) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 128;
  config.main_memory_capacity = 128;

  registry::KernelRegistry reg(config);
  auto key = makeKey(1);
  reg.registerKernel(key, makeMetadata("lib1", "func1"));
  ASSERT_NE(reg.lookup(key), nullptr);

  auto &nodes = reg.mainMemoryNodesForTest();
  auto node_it = nodes.find(key);
  ASSERT_NE(node_it, nodes.end());

  auto *node_ptr = node_it->second.get();
  ASSERT_NE(node_ptr, nullptr);

  auto before_buckets = nodes.bucket_count();
  nodes.rehash(before_buckets * 2 + 1);
  EXPECT_GT(nodes.bucket_count(), before_buckets);

  auto node_it_after = nodes.find(key);
  ASSERT_NE(node_it_after, nodes.end());
  EXPECT_EQ(node_it_after->second.get(), node_ptr);
  EXPECT_NE(reg.lookup(key), nullptr);
}
#endif

// ============================================================
// Register and lookup tests (Demand Paging)
// ============================================================

TEST(KernelRegistryTest, RegisterAddsToSecondaryStorage) {
  registry::KernelRegistry reg;
  auto key = makeKey(1);
  auto metadata = makeMetadata("lib1", "func1");

  reg.registerKernel(key, std::move(metadata));

  // Should be in secondary storage, not in cache or main memory
  EXPECT_EQ(reg.secondaryStorageSize(), 1u);
  EXPECT_EQ(reg.mainMemorySize(), 0u);
  EXPECT_EQ(reg.cacheSize(), 0u);
  EXPECT_TRUE(reg.contains(key));
}

TEST(KernelRegistryTest, LookupPromotesFromSecondary) {
  registry::KernelRegistry reg;
  auto key = makeKey(1);
  reg.registerKernel(key, makeMetadata("lib1", "func1"));

  // Before lookup: only in secondary
  EXPECT_EQ(reg.secondaryStorageSize(), 1u);
  EXPECT_EQ(reg.mainMemorySize(), 0u);

  // Lookup triggers rebuild and promotion
  auto *result = reg.lookup(key);

  EXPECT_NE(result, nullptr);
  // After lookup: promoted to main memory and cache
  EXPECT_EQ(reg.secondaryStorageSize(), 0u);
  EXPECT_EQ(reg.mainMemorySize(), 1u);
  EXPECT_EQ(reg.cacheSize(), 1u);
  EXPECT_EQ(reg.stats().secondary_hits, 1u);
}

TEST(KernelRegistryTest, LookupMiss) {
  registry::KernelRegistry reg;
  auto key = makeKey(999);

  auto *result = reg.lookup(key);

  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(reg.stats().misses, 1u);
}

TEST(KernelRegistryTest, CacheHitAfterPromotion) {
  registry::KernelRegistry reg;
  auto key = makeKey(1);
  reg.registerKernel(key, makeMetadata("lib1", "func1"));

  // First lookup promotes from secondary
  reg.lookup(key);
  EXPECT_EQ(reg.stats().secondary_hits, 1u);

  // Second lookup should hit cache
  reg.lookup(key);
  EXPECT_EQ(reg.stats().cache_hits, 1u);
}

// ============================================================
// Cache eviction tests
// ============================================================

TEST(KernelRegistryTest, CacheEviction) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 2;
  config.main_memory_capacity = 10;
  registry::KernelRegistry reg(config);

  // Register 3 kernels
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.registerKernel(makeKey(2), makeMetadata("lib2", "func2"));
  reg.registerKernel(makeKey(3), makeMetadata("lib3", "func3"));

  // Lookup all (promotes to main memory and cache)
  reg.lookup(makeKey(1));
  reg.lookup(makeKey(2));
  reg.lookup(makeKey(3));

  EXPECT_EQ(reg.mainMemorySize(), 3u);
  EXPECT_EQ(reg.cacheSize(), 2u); // Only 2 in cache (capacity)

  // All should still be accessible
  EXPECT_NE(reg.lookup(makeKey(1)), nullptr);
  EXPECT_NE(reg.lookup(makeKey(2)), nullptr);
  EXPECT_NE(reg.lookup(makeKey(3)), nullptr);
}

// ============================================================
// Main Memory eviction tests
// ============================================================

TEST(KernelRegistryTest, MainMemoryEviction) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 2;
  config.main_memory_capacity = 2;
  registry::KernelRegistry reg(config);

  // Register 3 kernels
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.registerKernel(makeKey(2), makeMetadata("lib2", "func2"));
  reg.registerKernel(makeKey(3), makeMetadata("lib3", "func3"));

  // Lookup all (causes eviction on 3rd)
  reg.lookup(makeKey(1));
  reg.lookup(makeKey(2));
  reg.lookup(makeKey(3)); // Should evict key1 from main memory

  EXPECT_EQ(reg.mainMemorySize(), 2u);
  EXPECT_EQ(reg.secondaryStorageSize(), 1u); // One demoted back
}

// ============================================================
// LRU ordering tests
// ============================================================

TEST(KernelRegistryTest, LruOrderingOnAccess) {
  registry::KernelRegistryConfig config;
  config.cache_capacity = 2;
  config.main_memory_capacity = 3;
  registry::KernelRegistry reg(config);

  // Register and lookup 3 kernels
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.registerKernel(makeKey(2), makeMetadata("lib2", "func2"));
  reg.registerKernel(makeKey(3), makeMetadata("lib3", "func3"));
  reg.lookup(makeKey(1));
  reg.lookup(makeKey(2));
  reg.lookup(makeKey(3));

  // Access key1 to make it MRU
  reg.lookup(makeKey(1));

  // Register and lookup 4th - should evict key2 (LRU)
  reg.registerKernel(makeKey(4), makeMetadata("lib4", "func4"));
  reg.lookup(makeKey(4));

  EXPECT_EQ(reg.mainMemorySize(), 3u);
  EXPECT_EQ(reg.secondaryStorageSize(), 1u);

  // Key1 should still be accessible (was accessed recently)
  EXPECT_NE(reg.lookup(makeKey(1)), nullptr);
}

// ============================================================
// Clear and flush tests
// ============================================================

TEST(KernelRegistryTest, Clear) {
  registry::KernelRegistry reg;
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.registerKernel(makeKey(2), makeMetadata("lib2", "func2"));

  reg.clear();

  EXPECT_EQ(reg.cacheSize(), 0u);
  EXPECT_EQ(reg.mainMemorySize(), 0u);
  EXPECT_EQ(reg.secondaryStorageSize(), 0u);
  EXPECT_FALSE(reg.contains(makeKey(1)));
}

TEST(KernelRegistryTest, Flush) {
  registry::KernelRegistry reg;
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.lookup(makeKey(1)); // Promote to cache

  EXPECT_EQ(reg.cacheSize(), 1u);
  EXPECT_EQ(reg.mainMemorySize(), 1u);

  reg.flush();

  EXPECT_EQ(reg.cacheSize(), 0u);
  EXPECT_EQ(reg.mainMemorySize(), 1u); // Still in main memory
}

// ============================================================
// Prefetch tests
// ============================================================

TEST(KernelRegistryTest, Prefetch) {
  registry::KernelRegistry reg;
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));

  // Before prefetch: only in secondary
  EXPECT_EQ(reg.secondaryStorageSize(), 1u);
  EXPECT_EQ(reg.cacheSize(), 0u);

  bool result = reg.prefetch(makeKey(1));

  EXPECT_TRUE(result);
  EXPECT_EQ(reg.cacheSize(), 1u);
  EXPECT_EQ(reg.secondaryStorageSize(), 0u);
}

TEST(KernelRegistryTest, PrefetchNotFound) {
  registry::KernelRegistry reg;

  bool result = reg.prefetch(makeKey(999));

  EXPECT_FALSE(result);
}

// ============================================================
// Contains tests
// ============================================================

TEST(KernelRegistryTest, ContainsInSecondary) {
  registry::KernelRegistry reg;
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));

  EXPECT_TRUE(reg.contains(makeKey(1)));
  EXPECT_FALSE(reg.contains(makeKey(2)));
}

TEST(KernelRegistryTest, ContainsInMainMemory) {
  registry::KernelRegistry reg;
  reg.registerKernel(makeKey(1), makeMetadata("lib1", "func1"));
  reg.lookup(makeKey(1));

  EXPECT_TRUE(reg.contains(makeKey(1)));
}

// ============================================================
// Stats tests
// ============================================================

TEST(KernelRegistryTest, StatsTracking) {
  registry::KernelRegistry reg;
  auto key = makeKey(1);
  reg.registerKernel(key, makeMetadata("lib1", "func1"));

  // Secondary hit on first lookup
  reg.lookup(key);
  EXPECT_EQ(reg.stats().secondary_hits, 1u);

  // Cache hit on second lookup
  reg.lookup(key);
  EXPECT_EQ(reg.stats().cache_hits, 1u);

  // Miss for unknown key
  reg.lookup(makeKey(999));
  EXPECT_EQ(reg.stats().misses, 1u);
}

} // namespace
