#include "orteaf/internal/kernel/param/param.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kernel = orteaf::internal::kernel;
using kernel::ArrayView;
using kernel::Param;
using kernel::ParamId;

// ============================================================
// Param construction tests
// ============================================================

TEST(Param, ConstructWithFloat) {
  Param p{ParamId::Alpha, 1.5f};
  EXPECT_EQ(p.id(), ParamId::Alpha);

  auto *val = p.tryGet<float>();
  ASSERT_NE(val, nullptr);
  EXPECT_FLOAT_EQ(*val, 1.5f);
}

TEST(Param, ConstructWithDouble) {
  Param p{ParamId::Scale, 2.5};
  EXPECT_EQ(p.id(), ParamId::Scale);

  auto *val = p.tryGet<double>();
  ASSERT_NE(val, nullptr);
  EXPECT_DOUBLE_EQ(*val, 2.5);
}

TEST(Param, ConstructWithInt) {
  Param p{ParamId::Axis, 42};
  EXPECT_EQ(p.id(), ParamId::Axis);

  auto *val = p.tryGet<int>();
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 42);
}

TEST(Param, ConstructWithSizeT) {
  Param p{ParamId::Dim, std::size_t{128}};
  EXPECT_EQ(p.id(), ParamId::Dim);

  auto *val = p.tryGet<std::size_t>();
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 128);
}

TEST(Param, ConstructWithVoidPtr) {
  int dummy = 0;
  void *ptr = &dummy;
  Param p{ParamId::DataPtr, ptr};
  EXPECT_EQ(p.id(), ParamId::DataPtr);

  auto *val = p.tryGet<void *>();
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, ptr);
}

// ============================================================
// ArrayView tests
// ============================================================

TEST(Param, ConstructWithArrayViewInt) {
  std::array<int, 3> arr = {1, 2, 3};
  ArrayView<const int> view{arr.data(), arr.size()};
  Param p{ParamId::Axis, view};

  auto *retrieved = p.tryGet<ArrayView<const int>>();
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->count, 3);
  EXPECT_EQ(retrieved->data, arr.data());
}

TEST(Param, ConstructWithArrayViewSizeT) {
  std::vector<std::size_t> vec = {10, 20, 30, 40};
  ArrayView<const std::size_t> view{vec.data(), vec.size()};
  Param p{ParamId::Dim, view};

  auto *retrieved = p.tryGet<ArrayView<const std::size_t>>();
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->count, 4);
  EXPECT_EQ(retrieved->data, vec.data());
}

TEST(Param, ConstructWithArrayViewFloat) {
  float arr[] = {1.0f, 2.0f, 3.0f};
  ArrayView<const float> view{arr, 3};
  Param p{ParamId::Alpha, view};

  auto *retrieved = p.tryGet<ArrayView<const float>>();
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->count, 3);
  EXPECT_EQ(retrieved->data, arr);
}

TEST(Param, ConstructWithArrayViewDouble) {
  double arr[] = {1.0, 2.0};
  ArrayView<const double> view{arr, 2};
  Param p{ParamId::Scale, view};

  auto *retrieved = p.tryGet<ArrayView<const double>>();
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->count, 2);
  EXPECT_EQ(retrieved->data, arr);
}

// ============================================================
// Type safety tests
// ============================================================

TEST(Param, TryGetWrongTypeReturnsNull) {
  Param p{ParamId::Alpha, 1.5f};
  EXPECT_EQ(p.tryGet<double>(), nullptr);
  EXPECT_EQ(p.tryGet<int>(), nullptr);
  EXPECT_EQ(p.tryGet<void *>(), nullptr);
}

TEST(Param, TryGetCorrectTypeSucceeds) {
  Param p{ParamId::Alpha, 1.5f};
  EXPECT_NE(p.tryGet<float>(), nullptr);
}

// ============================================================
// Visitor pattern tests
// ============================================================

TEST(Param, VisitFloat) {
  Param p{ParamId::Alpha, 2.5f};

  bool visited_float = false;
  p.visit([&](auto &&val) {
    using T = std::decay_t<decltype(val)>;
    if constexpr (std::is_same_v<T, float>) {
      visited_float = true;
      EXPECT_FLOAT_EQ(val, 2.5f);
    }
  });

  EXPECT_TRUE(visited_float);
}

TEST(Param, VisitArrayView) {
  std::array<int, 2> arr = {10, 20};
  ArrayView<const int> view{arr.data(), arr.size()};
  Param p{ParamId::Axis, view};

  bool visited_array = false;
  p.visit([&](auto &&val) {
    using T = std::decay_t<decltype(val)>;
    if constexpr (std::is_same_v<T, ArrayView<const int>>) {
      visited_array = true;
      EXPECT_EQ(val.count, 2);
    }
  });

  EXPECT_TRUE(visited_array);
}

// ============================================================
// Equality and hash tests
// ============================================================

TEST(Param, EqualityComparison) {
  Param p1{ParamId::Alpha, 1.5f};
  Param p2{ParamId::Alpha, 1.5f};
  Param p3{ParamId::Alpha, 2.5f};
  Param p4{ParamId::Beta, 1.5f};

  EXPECT_EQ(p1, p2);
  EXPECT_NE(p1, p3); // Different value
  EXPECT_NE(p1, p4); // Different ID
}

TEST(Param, HashSupport) {
  Param p1{ParamId::Alpha, 1.5f};
  Param p2{ParamId::Alpha, 1.5f};
  Param p3{ParamId::Beta, 1.5f};

  std::hash<Param> hasher;
  EXPECT_EQ(hasher(p1), hasher(p2));
  // Different IDs should likely have different hashes (not guaranteed, but
  // likely)
  EXPECT_NE(hasher(p1), hasher(p3));
}

TEST(Param, UnorderedSetUsage) {
  std::unordered_set<Param> param_set;

  Param p1{ParamId::Alpha, 1.0f};
  Param p2{ParamId::Beta, 2.0f};
  Param p3{ParamId::Alpha, 1.0f}; // Duplicate of p1

  param_set.insert(p1);
  param_set.insert(p2);
  param_set.insert(p3);

  EXPECT_EQ(param_set.size(), 2);
}

TEST(Param, UnorderedMapUsage) {
  std::unordered_map<Param, std::string> param_map;

  Param p1{ParamId::Alpha, 1.0f};
  Param p2{ParamId::Beta, 2.0f};

  param_map[p1] = "alpha";
  param_map[p2] = "beta";

  EXPECT_EQ(param_map.size(), 2);
  EXPECT_EQ(param_map[p1], "alpha");
  EXPECT_EQ(param_map[p2], "beta");
}

// ============================================================
// Practical use case tests
// ============================================================

TEST(Param, HeterogeneousVector) {
  std::vector<Param> params;

  // Mix of different types
  params.emplace_back(ParamId::Alpha, 1.0f);
  params.emplace_back(ParamId::Scale, 2.0);
  params.emplace_back(ParamId::Axis, 3);
  params.emplace_back(ParamId::Dim, std::size_t{4});

  int dummy = 0;
  params.emplace_back(ParamId::DataPtr, static_cast<void *>(&dummy));

  std::array<int, 2> axes = {0, 1};
  params.emplace_back(ParamId::Axis,
                      ArrayView<const int>{axes.data(), axes.size()});

  EXPECT_EQ(params.size(), 6);

  // Verify each type
  EXPECT_NE(params[0].tryGet<float>(), nullptr);
  EXPECT_NE(params[1].tryGet<double>(), nullptr);
  EXPECT_NE(params[2].tryGet<int>(), nullptr);
  EXPECT_NE(params[3].tryGet<std::size_t>(), nullptr);
  EXPECT_NE(params[4].tryGet<void *>(), nullptr);
  EXPECT_NE(params[5].tryGet<ArrayView<const int>>(), nullptr);
}

TEST(Param, MutableAccess) {
  Param p{ParamId::Alpha, 1.0f};

  // Mutable access via tryGet
  auto *val = p.tryGet<float>();
  ASSERT_NE(val, nullptr);
  *val = 2.0f;

  // Verify mutation
  auto *val2 = p.tryGet<float>();
  ASSERT_NE(val2, nullptr);
  EXPECT_FLOAT_EQ(*val2, 2.0f);
}
