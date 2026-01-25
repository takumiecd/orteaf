#include "orteaf/internal/kernel/core/layout.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

namespace kernel = orteaf::internal::kernel;

// ============================================================
// Layout basic functionality tests
// ============================================================

TEST(Layout, DefaultConstructedIsZero) {
  kernel::Layout layout{};
  EXPECT_EQ(static_cast<std::uint64_t>(layout), 0);
}

TEST(Layout, ExplicitConstructionWithValue) {
  kernel::Layout layout{static_cast<kernel::Layout>(1)};
  EXPECT_EQ(static_cast<std::uint64_t>(layout), 1);
}

TEST(Layout, StaticCastFromInteger) {
  auto layout = static_cast<kernel::Layout>(2);
  EXPECT_EQ(static_cast<std::uint64_t>(layout), 2);
}

TEST(Layout, StaticCastToInteger) {
  kernel::Layout layout = static_cast<kernel::Layout>(3);
  std::uint64_t value = static_cast<std::uint64_t>(layout);
  EXPECT_EQ(value, 3);
}

// ============================================================
// Layout comparison operator tests
// ============================================================

TEST(Layout, EqualityOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(1);
  kernel::Layout layout3 = static_cast<kernel::Layout>(2);

  EXPECT_TRUE(layout1 == layout2);
  EXPECT_FALSE(layout1 == layout3);
}

TEST(Layout, InequalityOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(1);
  kernel::Layout layout3 = static_cast<kernel::Layout>(2);

  EXPECT_FALSE(layout1 != layout2);
  EXPECT_TRUE(layout1 != layout3);
}

TEST(Layout, LessThanOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);

  EXPECT_TRUE(layout1 < layout2);
  EXPECT_FALSE(layout2 < layout1);
  EXPECT_FALSE(layout1 < layout1);
}

TEST(Layout, LessThanOrEqualOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);
  kernel::Layout layout3 = static_cast<kernel::Layout>(1);

  EXPECT_TRUE(layout1 <= layout2);
  EXPECT_TRUE(layout1 <= layout3);
  EXPECT_FALSE(layout2 <= layout1);
}

TEST(Layout, GreaterThanOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);

  EXPECT_TRUE(layout2 > layout1);
  EXPECT_FALSE(layout1 > layout2);
  EXPECT_FALSE(layout1 > layout1);
}

TEST(Layout, GreaterThanOrEqualOperator) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);
  kernel::Layout layout3 = static_cast<kernel::Layout>(1);

  EXPECT_TRUE(layout2 >= layout1);
  EXPECT_TRUE(layout1 >= layout3);
  EXPECT_FALSE(layout1 >= layout2);
}

// ============================================================
// Layout hash support tests
// ============================================================

TEST(Layout, HashSupport) {
  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(1);
  kernel::Layout layout3 = static_cast<kernel::Layout>(2);

  std::hash<kernel::Layout> hasher;

  // Same values should produce same hash
  EXPECT_EQ(hasher(layout1), hasher(layout2));

  // Different values should (very likely) produce different hashes
  EXPECT_NE(hasher(layout1), hasher(layout3));
}

TEST(Layout, UnorderedSetUsage) {
  std::unordered_set<kernel::Layout> layout_set;

  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);
  kernel::Layout layout3 = static_cast<kernel::Layout>(1); // Same as layout1

  layout_set.insert(layout1);
  layout_set.insert(layout2);
  layout_set.insert(layout3); // Should not increase size

  EXPECT_EQ(layout_set.size(), 2);
  EXPECT_TRUE(layout_set.count(layout1) > 0);
  EXPECT_TRUE(layout_set.count(layout2) > 0);
  EXPECT_TRUE(layout_set.count(layout3) > 0); // Same as layout1
}

TEST(Layout, UnorderedMapUsage) {
  std::unordered_map<kernel::Layout, std::string> layout_map;

  kernel::Layout layout1 = static_cast<kernel::Layout>(1);
  kernel::Layout layout2 = static_cast<kernel::Layout>(2);

  layout_map[layout1] = "RowMajor";
  layout_map[layout2] = "ColumnMajor";

  EXPECT_EQ(layout_map.size(), 2);
  EXPECT_EQ(layout_map[layout1], "RowMajor");
  EXPECT_EQ(layout_map[layout2], "ColumnMajor");

  // Overwrite existing key
  layout_map[layout1] = "Updated RowMajor";
  EXPECT_EQ(layout_map.size(), 2);
  EXPECT_EQ(layout_map[layout1], "Updated RowMajor");
}

// ============================================================
// Layout constexpr tests
// ============================================================

TEST(Layout, ConstexprSupport) {
  constexpr kernel::Layout layout1{};
  static_assert(static_cast<std::uint64_t>(layout1) == 0);

  constexpr kernel::Layout layout2 = static_cast<kernel::Layout>(1);
  static_assert(static_cast<std::uint64_t>(layout2) == 1);

  constexpr kernel::Layout layout3 = static_cast<kernel::Layout>(1);
  static_assert(layout2 == layout3);
  static_assert(!(layout2 != layout3));
}

// ============================================================
// Layout for memory pattern examples
// ============================================================

TEST(Layout, MemoryLayoutExample) {
  // Example: Different memory layout patterns
  kernel::Layout row_major = static_cast<kernel::Layout>(0);
  kernel::Layout column_major = static_cast<kernel::Layout>(1);
  kernel::Layout blocked = static_cast<kernel::Layout>(2);

  EXPECT_NE(row_major, column_major);
  EXPECT_NE(column_major, blocked);
  EXPECT_TRUE(row_major < column_major);
  EXPECT_TRUE(column_major < blocked);
}
