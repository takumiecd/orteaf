#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <orteaf/src/extension/kernel/common/layout_common.h>

namespace common_layout = ::orteaf::extension::kernel::common::layout;

namespace {

common_layout::ShapeVector makeShapeVector(
    std::initializer_list<std::int64_t> values) {
  common_layout::ShapeVector vec{};
  vec.size = static_cast<std::uint8_t>(values.size());
  std::size_t index = 0;
  for (const auto value : values) {
    vec.data[index++] = value;
  }
  return vec;
}

TEST(LayoutCommonTest, AnalyzeLayoutThrowsOnShapeStrideRankMismatch) {
  const auto shape = makeShapeVector({2, 3});
  const auto strides = makeShapeVector({3});

  EXPECT_ANY_THROW(
      common_layout::analyzeLayout(shape, strides, 0, "Test kernel", "input"));
}

TEST(LayoutCommonTest, AnalyzeLayoutThrowsOnNegativeShapeDimension) {
  const auto shape = makeShapeVector({2, -1});
  const auto strides = makeShapeVector({3, 1});

  EXPECT_ANY_THROW(
      common_layout::analyzeLayout(shape, strides, 0, "Test kernel", "input"));
}

TEST(LayoutCommonTest, AnalyzeLayoutDetectsZeroDimension) {
  const auto shape = makeShapeVector({4, 0, 2});
  const auto strides = makeShapeVector({2, 2, 1});

  const auto info =
      common_layout::analyzeLayout(shape, strides, 0, "Test kernel", "input");

  EXPECT_TRUE(info.has_zero);
  EXPECT_EQ(info.numel, 0u);
}

TEST(LayoutCommonTest, AnalyzeLayoutComputesContiguousFlag) {
  const auto shape = makeShapeVector({2, 3});
  const auto contiguous_strides = makeShapeVector({3, 1});
  const auto non_contiguous_strides = makeShapeVector({4, 1});

  const auto contiguous_info = common_layout::analyzeLayout(
      shape, contiguous_strides, 5, "Test kernel", "input");
  const auto non_contiguous_info = common_layout::analyzeLayout(
      shape, non_contiguous_strides, 5, "Test kernel", "input");

  EXPECT_TRUE(contiguous_info.contiguous);
  EXPECT_FALSE(non_contiguous_info.contiguous);
}

TEST(LayoutCommonTest, AnalyzeLayoutTracksNegativeStrideBounds) {
  const auto shape = makeShapeVector({2, 3});
  const auto strides = makeShapeVector({-4, 1});

  const auto info =
      common_layout::analyzeLayout(shape, strides, 10, "Test kernel", "input");

  EXPECT_EQ(info.min_index, 6);
  EXPECT_EQ(info.max_index, 12);
}

TEST(LayoutCommonTest, AnalyzeLayoutThrowsOnIndexOverflow) {
  const auto shape = makeShapeVector({2});
  const auto strides = makeShapeVector({std::numeric_limits<std::int64_t>::max()});

  EXPECT_ANY_THROW(
      common_layout::analyzeLayout(shape, strides, 1, "Test kernel", "input"));
}

TEST(LayoutCommonTest, FillFillLayoutParamsValidatesRanges) {
  common_layout::FillLayoutParams params{};
  const auto too_large_dim = static_cast<std::int64_t>(
                                 std::numeric_limits<std::uint32_t>::max()) +
                             1;

  EXPECT_ANY_THROW(common_layout::fillFillLayoutParams(
      params, makeShapeVector({too_large_dim}), makeShapeVector({1}),
      "Test kernel"));
  EXPECT_ANY_THROW(common_layout::fillFillLayoutParams(
      params, makeShapeVector({2}),
      makeShapeVector({static_cast<std::int64_t>(
          std::numeric_limits<std::int32_t>::max()) +
                       1}),
      "Test kernel"));
}

TEST(LayoutCommonTest, FillTransferLayoutParamsValidatesRanges) {
  common_layout::TransferLayoutParams params{};
  const auto too_large_dim = static_cast<std::int64_t>(
                                 std::numeric_limits<std::uint32_t>::max()) +
                             1;

  EXPECT_ANY_THROW(common_layout::fillTransferLayoutParams(
      params, makeShapeVector({too_large_dim}), makeShapeVector({1}),
      makeShapeVector({1}), "Test kernel"));
  EXPECT_ANY_THROW(common_layout::fillTransferLayoutParams(
      params, makeShapeVector({2}), makeShapeVector({1}),
      makeShapeVector({static_cast<std::int64_t>(
          std::numeric_limits<std::int32_t>::max()) +
                       1}),
      "Test kernel"));
}

} // namespace
