#include "orteaf/internal/kernel/variant.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

namespace kernel = orteaf::internal::kernel;

// ============================================================
// Variant basic functionality tests
// ============================================================

TEST(Variant, DefaultConstructedIsZero) {
  kernel::Variant variant{};
  EXPECT_EQ(static_cast<std::uint64_t>(variant), 0);
}

TEST(Variant, ExplicitConstructionWithValue) {
  kernel::Variant variant{static_cast<kernel::Variant>(1)};
  EXPECT_EQ(static_cast<std::uint64_t>(variant), 1);
}

TEST(Variant, StaticCastFromInteger) {
  auto variant = static_cast<kernel::Variant>(2);
  EXPECT_EQ(static_cast<std::uint64_t>(variant), 2);
}

TEST(Variant, StaticCastToInteger) {
  kernel::Variant variant = static_cast<kernel::Variant>(3);
  std::uint64_t value = static_cast<std::uint64_t>(variant);
  EXPECT_EQ(value, 3);
}

// ============================================================
// Variant comparison operator tests
// ============================================================

TEST(Variant, EqualityOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(1);
  kernel::Variant variant3 = static_cast<kernel::Variant>(2);

  EXPECT_TRUE(variant1 == variant2);
  EXPECT_FALSE(variant1 == variant3);
}

TEST(Variant, InequalityOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(1);
  kernel::Variant variant3 = static_cast<kernel::Variant>(2);

  EXPECT_FALSE(variant1 != variant2);
  EXPECT_TRUE(variant1 != variant3);
}

TEST(Variant, LessThanOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);

  EXPECT_TRUE(variant1 < variant2);
  EXPECT_FALSE(variant2 < variant1);
  EXPECT_FALSE(variant1 < variant1);
}

TEST(Variant, LessThanOrEqualOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);
  kernel::Variant variant3 = static_cast<kernel::Variant>(1);

  EXPECT_TRUE(variant1 <= variant2);
  EXPECT_TRUE(variant1 <= variant3);
  EXPECT_FALSE(variant2 <= variant1);
}

TEST(Variant, GreaterThanOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);

  EXPECT_TRUE(variant2 > variant1);
  EXPECT_FALSE(variant1 > variant2);
  EXPECT_FALSE(variant1 > variant1);
}

TEST(Variant, GreaterThanOrEqualOperator) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);
  kernel::Variant variant3 = static_cast<kernel::Variant>(1);

  EXPECT_TRUE(variant2 >= variant1);
  EXPECT_TRUE(variant1 >= variant3);
  EXPECT_FALSE(variant1 >= variant2);
}

// ============================================================
// Variant hash support tests
// ============================================================

TEST(Variant, HashSupport) {
  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(1);
  kernel::Variant variant3 = static_cast<kernel::Variant>(2);

  std::hash<kernel::Variant> hasher;

  // Same values should produce same hash
  EXPECT_EQ(hasher(variant1), hasher(variant2));

  // Different values should (very likely) produce different hashes
  EXPECT_NE(hasher(variant1), hasher(variant3));
}

TEST(Variant, UnorderedSetUsage) {
  std::unordered_set<kernel::Variant> variant_set;

  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);
  kernel::Variant variant3 =
      static_cast<kernel::Variant>(1); // Same as variant1

  variant_set.insert(variant1);
  variant_set.insert(variant2);
  variant_set.insert(variant3); // Should not increase size

  EXPECT_EQ(variant_set.size(), 2);
  EXPECT_TRUE(variant_set.count(variant1) > 0);
  EXPECT_TRUE(variant_set.count(variant2) > 0);
  EXPECT_TRUE(variant_set.count(variant3) > 0); // Same as variant1
}

TEST(Variant, UnorderedMapUsage) {
  std::unordered_map<kernel::Variant, std::string> variant_map;

  kernel::Variant variant1 = static_cast<kernel::Variant>(1);
  kernel::Variant variant2 = static_cast<kernel::Variant>(2);

  variant_map[variant1] = "Optimization Level 1";
  variant_map[variant2] = "Optimization Level 2";

  EXPECT_EQ(variant_map.size(), 2);
  EXPECT_EQ(variant_map[variant1], "Optimization Level 1");
  EXPECT_EQ(variant_map[variant2], "Optimization Level 2");

  // Overwrite existing key
  variant_map[variant1] = "Updated Level 1";
  EXPECT_EQ(variant_map.size(), 2);
  EXPECT_EQ(variant_map[variant1], "Updated Level 1");
}

// ============================================================
// Variant constexpr tests
// ============================================================

TEST(Variant, ConstexprSupport) {
  constexpr kernel::Variant variant1{};
  static_assert(static_cast<std::uint64_t>(variant1) == 0);

  constexpr kernel::Variant variant2 = static_cast<kernel::Variant>(1);
  static_assert(static_cast<std::uint64_t>(variant2) == 1);

  constexpr kernel::Variant variant3 = static_cast<kernel::Variant>(1);
  static_assert(variant2 == variant3);
  static_assert(!(variant2 != variant3));
}

// ============================================================
// Variant for optimization levels
// ============================================================

TEST(Variant, OptimizationLevelExample) {
  // Example: Different optimization levels for the same operation
  kernel::Variant baseline = static_cast<kernel::Variant>(0);
  kernel::Variant optimized = static_cast<kernel::Variant>(1);
  kernel::Variant highly_optimized = static_cast<kernel::Variant>(2);

  EXPECT_NE(baseline, optimized);
  EXPECT_NE(optimized, highly_optimized);
  EXPECT_TRUE(baseline < optimized);
  EXPECT_TRUE(optimized < highly_optimized);
}
