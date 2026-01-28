#include "orteaf/internal/kernel/param/param_id.h"

#include <gtest/gtest.h>

#include <orteaf/kernel/param_id_tables.h>

#include <unordered_map>
#include <unordered_set>

namespace kernel = orteaf::internal::kernel;
namespace param_tables = orteaf::generated::param_id_tables;

// ============================================================
// ParamId basic functionality tests
// ============================================================

TEST(ParamId, DefaultConstructedIsZero) {
  kernel::ParamId param_id{};
  EXPECT_EQ(static_cast<std::uint64_t>(param_id), 0);
}

TEST(ParamId, ExplicitConstructionWithValue) {
  kernel::ParamId param_id{static_cast<kernel::ParamId>(42)};
  EXPECT_EQ(static_cast<std::uint64_t>(param_id), 42);
}

TEST(ParamId, StaticCastFromInteger) {
  auto param_id = static_cast<kernel::ParamId>(123);
  EXPECT_EQ(static_cast<std::uint64_t>(param_id), 123);
}

TEST(ParamId, StaticCastToInteger) {
  kernel::ParamId param_id = static_cast<kernel::ParamId>(456);
  std::uint64_t value = static_cast<std::uint64_t>(param_id);
  EXPECT_EQ(value, 456);
}

// ============================================================
// ParamId enum values from generated code
// ============================================================

TEST(ParamId, EnumValuesAreDistinct) {
  // Test that different ParamIds have different values
  kernel::ParamId alpha = kernel::ParamId::Alpha;
  kernel::ParamId beta = kernel::ParamId::Beta;
  kernel::ParamId scale = kernel::ParamId::Scale;

  EXPECT_NE(alpha, beta);
  EXPECT_NE(beta, scale);
  EXPECT_NE(alpha, scale);
}

TEST(ParamId, SameEnumValuesAreEqual) {
  // Test that same ParamId values are equal
  kernel::ParamId alpha1 = kernel::ParamId::Alpha;
  kernel::ParamId alpha2 = kernel::ParamId::Alpha;

  EXPECT_EQ(alpha1, alpha2);
}

// ============================================================
// ParamId comparison operator tests
// ============================================================

TEST(ParamId, EqualityOperator) {
  kernel::ParamId param_id1 = kernel::ParamId::Alpha;
  kernel::ParamId param_id2 = kernel::ParamId::Alpha;
  kernel::ParamId param_id3 = kernel::ParamId::Beta;

  EXPECT_TRUE(param_id1 == param_id2);
  EXPECT_FALSE(param_id1 == param_id3);
}

TEST(ParamId, InequalityOperator) {
  kernel::ParamId param_id1 = kernel::ParamId::Alpha;
  kernel::ParamId param_id2 = kernel::ParamId::Alpha;
  kernel::ParamId param_id3 = kernel::ParamId::Beta;

  EXPECT_FALSE(param_id1 != param_id2);
  EXPECT_TRUE(param_id1 != param_id3);
}

// ============================================================
// ParamId hash support tests
// ============================================================

TEST(ParamId, HashSupport) {
  kernel::ParamId param_id1 = kernel::ParamId::Alpha;
  kernel::ParamId param_id2 = kernel::ParamId::Alpha;
  kernel::ParamId param_id3 = kernel::ParamId::Beta;

  std::hash<kernel::ParamId> hasher;

  // Same values should produce same hash
  EXPECT_EQ(hasher(param_id1), hasher(param_id2));

  // Different values should (very likely) produce different hashes
  EXPECT_NE(hasher(param_id1), hasher(param_id3));
}

TEST(ParamId, UnorderedSetUsage) {
  std::unordered_set<kernel::ParamId> param_id_set;

  kernel::ParamId param_id1 = kernel::ParamId::Alpha;
  kernel::ParamId param_id2 = kernel::ParamId::Beta;
  kernel::ParamId param_id3 = kernel::ParamId::Alpha; // Same as param_id1

  param_id_set.insert(param_id1);
  param_id_set.insert(param_id2);
  param_id_set.insert(param_id3); // Should not increase size

  EXPECT_EQ(param_id_set.size(), 2);
  EXPECT_TRUE(param_id_set.count(param_id1) > 0);
  EXPECT_TRUE(param_id_set.count(param_id2) > 0);
  EXPECT_TRUE(param_id_set.count(param_id3) > 0); // Same as param_id1
}

TEST(ParamId, UnorderedMapUsage) {
  std::unordered_map<kernel::ParamId, std::string> param_id_map;

  kernel::ParamId param_id1 = kernel::ParamId::Alpha;
  kernel::ParamId param_id2 = kernel::ParamId::Beta;

  param_id_map[param_id1] = "Alpha scalar";
  param_id_map[param_id2] = "Beta scalar";

  EXPECT_EQ(param_id_map.size(), 2);
  EXPECT_EQ(param_id_map[param_id1], "Alpha scalar");
  EXPECT_EQ(param_id_map[param_id2], "Beta scalar");

  // Overwrite existing key
  param_id_map[param_id1] = "Updated Input";
  EXPECT_EQ(param_id_map.size(), 2);
  EXPECT_EQ(param_id_map[param_id1], "Updated Input");
}

// ============================================================
// ParamId constexpr tests
// ============================================================

TEST(ParamId, ConstexprSupport) {
  constexpr kernel::ParamId param_id1{};
  constexpr kernel::ParamId param_id2 = kernel::ParamId::Alpha;
  constexpr kernel::ParamId param_id3 = kernel::ParamId::Alpha;

  // Test constexpr equality
  static_assert(param_id2 == kernel::ParamId::Alpha);
  static_assert(!(param_id2 != kernel::ParamId::Alpha));
  static_assert(param_id2 == param_id3);
}

// ============================================================
// ParamId generated tables tests
// ============================================================

TEST(ParamIdTables, ParamIdCountCorrect) {
  EXPECT_EQ(param_tables::kParamIdCount, 13);
}

// ============================================================
// ParamInfo tests
// ============================================================

TEST(ParamInfo, InputHasCorrectTypeName) {
  using InputInfo = param_tables::ParamInfo<kernel::ParamId::Alpha>;
  EXPECT_EQ(InputInfo::kTypeName, "float");
  EXPECT_FALSE(InputInfo::kDescription.empty());
}

TEST(ParamInfo, AlphaHasFloatTypeName) {
  using AlphaInfo = param_tables::ParamInfo<kernel::ParamId::Alpha>;
  EXPECT_EQ(AlphaInfo::kTypeName, "float");
  EXPECT_FALSE(AlphaInfo::kDescription.empty());
}

TEST(ParamInfo, ScaleHasDoubleTypeName) {
  using ScaleInfo = param_tables::ParamInfo<kernel::ParamId::Scale>;
  EXPECT_EQ(ScaleInfo::kTypeName, "double");
}

TEST(ParamInfo, AxisHasIntTypeName) {
  using AxisInfo = param_tables::ParamInfo<kernel::ParamId::Axis>;
  EXPECT_EQ(AxisInfo::kTypeName, "int");
}

TEST(ParamInfo, DimHasSizeTTypeName) {
  using DimInfo = param_tables::ParamInfo<kernel::ParamId::Dim>;
  EXPECT_EQ(DimInfo::kTypeName, "std::size_t");
}

TEST(ParamInfo, BufferPtrHasVoidPtrTypeName) {
  using BufferPtrInfo = param_tables::ParamInfo<kernel::ParamId::BufferPtr>;
  EXPECT_EQ(BufferPtrInfo::kTypeName, "void*");
}

TEST(ParamInfo, AllTypeInfoHaveDescription) {
  // Test all ParamInfo have non-empty descriptions
  EXPECT_FALSE(param_tables::ParamInfo<kernel::ParamId::Alpha>::kDescription
                   .empty());
  EXPECT_FALSE(
      param_tables::ParamInfo<kernel::ParamId::Beta>::kDescription.empty());
  EXPECT_FALSE(param_tables::ParamInfo<kernel::ParamId::Alpha>::kDescription
                   .empty());
  EXPECT_FALSE(
      param_tables::ParamInfo<kernel::ParamId::Beta>::kDescription.empty());
  EXPECT_FALSE(
      param_tables::ParamInfo<kernel::ParamId::Axis>::kDescription.empty());
}

// ============================================================
// ParamId practical use case tests
// ============================================================

TEST(ParamId, PracticalKernelParameterRegistry) {
  // Simulate a parameter registry for kernel function signatures
  std::unordered_map<kernel::ParamId, std::string> param_registry;

  param_registry[kernel::ParamId::Alpha] = "alpha_val";
  param_registry[kernel::ParamId::Beta] = "beta_val";
  param_registry[kernel::ParamId::Scale] = "scale_val";

  EXPECT_EQ(param_registry.size(), 3);
  EXPECT_EQ(param_registry[kernel::ParamId::Alpha], "alpha_val");
  EXPECT_EQ(param_registry[kernel::ParamId::Scale], "scale_val");
}

TEST(ParamId, TypeNameLookup) {
  // Demonstrate type name lookup using ParamInfo
  using AlphaInfo = param_tables::ParamInfo<kernel::ParamId::Alpha>;
  using BetaInfo = param_tables::ParamInfo<kernel::ParamId::Beta>;

  // Both should be float
  EXPECT_EQ(AlphaInfo::kTypeName, "float");
  EXPECT_EQ(BetaInfo::kTypeName, "float");
}
