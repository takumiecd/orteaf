#include <array>
#include <system_error>

#include <gtest/gtest.h>

#include <orteaf/extension/tensor/layout/dense_tensor_layout.h>
#include <orteaf/internal/diagnostics/error/error.h>

namespace {

using Layout = ::orteaf::extension::tensor::DenseTensorLayout;
using Errc = ::orteaf::internal::diagnostics::error::OrteafErrc;

template <typename Fn>
void ExpectError(Errc expected, Fn &&fn) {
  try {
    fn();
    FAIL() << "Expected error";
  } catch (const std::system_error &ex) {
    EXPECT_EQ(ex.code(), ::orteaf::internal::diagnostics::error::makeErrorCode(
                               expected));
  }
}

TEST(DenseTensorLayoutTest, ContiguousBasics) {
  const std::array<Layout::Dim, 3> shape{2, 3, 4};
  Layout layout = Layout::contiguous(shape);

  EXPECT_EQ(layout.rank(), 3u);
  EXPECT_EQ(layout.shape().size(), 3u);
  EXPECT_EQ(layout.strides().size(), 3u);
  EXPECT_EQ(layout.shape()[0], 2);
  EXPECT_EQ(layout.shape()[1], 3);
  EXPECT_EQ(layout.shape()[2], 4);
  EXPECT_EQ(layout.strides()[0], 12);
  EXPECT_EQ(layout.strides()[1], 4);
  EXPECT_EQ(layout.strides()[2], 1);
  EXPECT_EQ(layout.offset(), 0);
  EXPECT_EQ(layout.numel(), 24);
  EXPECT_TRUE(layout.isContiguous());
}

TEST(DenseTensorLayoutTest, ContiguousScalarShape) {
  const std::array<Layout::Dim, 0> shape{};
  Layout layout = Layout::contiguous(shape);

  EXPECT_EQ(layout.rank(), 0u);
  EXPECT_EQ(layout.shape().size(), 0u);
  EXPECT_EQ(layout.strides().size(), 0u);
  EXPECT_EQ(layout.numel(), 1);
  EXPECT_TRUE(layout.isContiguous());
}

TEST(DenseTensorLayoutTest, ConstructorValidatesRank) {
  Layout::Dims shape;
  shape.pushBack(2);
  shape.pushBack(3);
  Layout::Dims strides;
  strides.pushBack(1);

  ExpectError(Errc::InvalidParameter,
              [&]() { Layout layout(shape, strides, 0); });
}

TEST(DenseTensorLayoutTest, ConstructorValidatesShape) {
  Layout::Dims shape;
  shape.pushBack(-1);
  Layout::Dims strides;
  strides.pushBack(1);

  ExpectError(Errc::InvalidParameter,
              [&]() { Layout layout(shape, strides, 0); });
}

TEST(DenseTensorLayoutTest, TransposePermutesShapeAndStrides) {
  const std::array<Layout::Dim, 3> shape{2, 3, 4};
  Layout layout = Layout::contiguous(shape);

  const std::array<std::size_t, 3> perm{2, 0, 1};
  Layout transposed = layout.transpose(perm);

  EXPECT_EQ(transposed.shape()[0], 4);
  EXPECT_EQ(transposed.shape()[1], 2);
  EXPECT_EQ(transposed.shape()[2], 3);
  EXPECT_EQ(transposed.strides()[0], 1);
  EXPECT_EQ(transposed.strides()[1], 12);
  EXPECT_EQ(transposed.strides()[2], 4);
  EXPECT_EQ(transposed.offset(), 0);
  EXPECT_FALSE(transposed.isContiguous());
}

TEST(DenseTensorLayoutTest, TransposeRejectsBadPerm) {
  const std::array<Layout::Dim, 2> shape{2, 3};
  Layout layout = Layout::contiguous(shape);

  const std::array<std::size_t, 1> bad_rank{0};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.transpose(bad_rank); });

  const std::array<std::size_t, 2> dup_perm{0, 0};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.transpose(dup_perm); });
}

TEST(DenseTensorLayoutTest, SliceUpdatesOffsetShapeAndStride) {
  const std::array<Layout::Dim, 2> shape{4, 5};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> starts{1, 2};
  const std::array<Layout::Dim, 2> sizes{2, 3};
  Layout sliced = layout.slice(starts, sizes);

  EXPECT_EQ(sliced.shape()[0], 2);
  EXPECT_EQ(sliced.shape()[1], 3);
  EXPECT_EQ(sliced.strides()[0], 5);
  EXPECT_EQ(sliced.strides()[1], 1);
  EXPECT_EQ(sliced.offset(), 7);
}

TEST(DenseTensorLayoutTest, SliceSupportsSteps) {
  const std::array<Layout::Dim, 1> shape{5};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 1> starts{4};
  const std::array<Layout::Dim, 1> sizes{3};
  const std::array<Layout::Dim, 1> steps{-1};
  Layout sliced = layout.slice(starts, sizes, steps);

  EXPECT_EQ(sliced.shape()[0], 3);
  EXPECT_EQ(sliced.strides()[0], -1);
  EXPECT_EQ(sliced.offset(), 4);
}

TEST(DenseTensorLayoutTest, SliceValidatesRanges) {
  const std::array<Layout::Dim, 1> shape{5};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 1> bad_start{5};
  const std::array<Layout::Dim, 1> size{1};
  ExpectError(Errc::OutOfRange,
              [&]() { (void)layout.slice(bad_start, size); });

  const std::array<Layout::Dim, 1> start{3};
  const std::array<Layout::Dim, 1> bad_size{3};
  ExpectError(Errc::OutOfRange,
              [&]() { (void)layout.slice(start, bad_size); });

  const std::array<Layout::Dim, 1> zero_step{0};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.slice(start, size, zero_step); });
}

TEST(DenseTensorLayoutTest, ReshapeUpdatesShapeAndStrides) {
  const std::array<Layout::Dim, 3> shape{2, 3, 4};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> new_shape{6, 4};
  Layout reshaped = layout.reshape(new_shape);

  EXPECT_EQ(reshaped.shape()[0], 6);
  EXPECT_EQ(reshaped.shape()[1], 4);
  EXPECT_EQ(reshaped.strides()[0], 4);
  EXPECT_EQ(reshaped.strides()[1], 1);
  EXPECT_EQ(reshaped.offset(), 0);
  EXPECT_TRUE(reshaped.isContiguous());
}

TEST(DenseTensorLayoutTest, ReshapeInfersDimension) {
  const std::array<Layout::Dim, 3> shape{2, 3, 4};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> new_shape{-1, 4};
  Layout reshaped = layout.reshape(new_shape);

  EXPECT_EQ(reshaped.shape()[0], 6);
  EXPECT_EQ(reshaped.shape()[1], 4);
}

TEST(DenseTensorLayoutTest, ReshapeRejectsInvalidRequests) {
  const std::array<Layout::Dim, 3> shape{2, 3, 4};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> bad_shape{-1, -1};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.reshape(bad_shape); });

  const std::array<Layout::Dim, 1> mismatch{5};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.reshape(mismatch); });

  const std::array<std::size_t, 3> perm{1, 0, 2};
  Layout non_contiguous = layout.transpose(perm);
  const std::array<Layout::Dim, 1> reshape_one{-1};
  ExpectError(Errc::InvalidState,
              [&]() { (void)non_contiguous.reshape(reshape_one); });
}

TEST(DenseTensorLayoutTest, ReshapeHandlesZero) {
  const std::array<Layout::Dim, 2> shape{0, 3};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> same_shape{0, 3};
  Layout reshaped = layout.reshape(same_shape);
  EXPECT_EQ(reshaped.numel(), 0);

  const std::array<Layout::Dim, 1> infer{-1};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.reshape(infer); });
}

TEST(DenseTensorLayoutTest, BroadcastExpandsStrides) {
  const std::array<Layout::Dim, 3> shape{3, 1, 5};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 3> expanded{3, 4, 5};
  Layout broadcasted = layout.broadcastTo(expanded);

  EXPECT_EQ(broadcasted.shape()[0], 3);
  EXPECT_EQ(broadcasted.shape()[1], 4);
  EXPECT_EQ(broadcasted.shape()[2], 5);
  EXPECT_EQ(broadcasted.strides()[0], 5);
  EXPECT_EQ(broadcasted.strides()[1], 0);
  EXPECT_EQ(broadcasted.strides()[2], 1);
}

TEST(DenseTensorLayoutTest, BroadcastAddsLeadingDims) {
  const std::array<Layout::Dim, 1> shape{5};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 2> expanded{2, 5};
  Layout broadcasted = layout.broadcastTo(expanded);

  EXPECT_EQ(broadcasted.shape()[0], 2);
  EXPECT_EQ(broadcasted.shape()[1], 5);
  EXPECT_EQ(broadcasted.strides()[0], 0);
  EXPECT_EQ(broadcasted.strides()[1], 1);
}

TEST(DenseTensorLayoutTest, BroadcastRejectsMismatch) {
  const std::array<Layout::Dim, 2> shape{2, 3};
  Layout layout = Layout::contiguous(shape);

  const std::array<Layout::Dim, 1> smaller{3};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.broadcastTo(smaller); });

  const std::array<Layout::Dim, 2> mismatch{2, 4};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.broadcastTo(mismatch); });
}

TEST(DenseTensorLayoutTest, SqueezeRemovesUnitDims) {
  const std::array<Layout::Dim, 4> shape{1, 3, 1, 4};
  Layout layout = Layout::contiguous(shape);

  Layout squeezed = layout.squeeze();
  EXPECT_EQ(squeezed.rank(), 2u);
  EXPECT_EQ(squeezed.shape()[0], 3);
  EXPECT_EQ(squeezed.shape()[1], 4);
  EXPECT_EQ(squeezed.strides()[0], 4);
  EXPECT_EQ(squeezed.strides()[1], 1);
}

TEST(DenseTensorLayoutTest, SqueezeWithDimsValidatesInput) {
  const std::array<Layout::Dim, 3> shape{1, 2, 3};
  Layout layout = Layout::contiguous(shape);

  const std::array<std::size_t, 1> non_unit{1};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.squeeze(non_unit); });

  const std::array<std::size_t, 1> out_of_range{4};
  ExpectError(Errc::OutOfRange,
              [&]() { (void)layout.squeeze(out_of_range); });

  const std::array<std::size_t, 2> dup{0, 0};
  ExpectError(Errc::InvalidParameter,
              [&]() { (void)layout.squeeze(dup); });
}

TEST(DenseTensorLayoutTest, UnsqueezeInsertsUnitDim) {
  const std::array<Layout::Dim, 2> shape{3, 4};
  Layout layout = Layout::contiguous(shape);

  Layout front = layout.unsqueeze(0);
  EXPECT_EQ(front.shape()[0], 1);
  EXPECT_EQ(front.shape()[1], 3);
  EXPECT_EQ(front.shape()[2], 4);
  EXPECT_EQ(front.strides()[0], 12);
  EXPECT_EQ(front.strides()[1], 4);
  EXPECT_EQ(front.strides()[2], 1);

  Layout back = layout.unsqueeze(2);
  EXPECT_EQ(back.shape()[0], 3);
  EXPECT_EQ(back.shape()[1], 4);
  EXPECT_EQ(back.shape()[2], 1);
  EXPECT_EQ(back.strides()[0], 4);
  EXPECT_EQ(back.strides()[1], 1);
  EXPECT_EQ(back.strides()[2], 1);
}

TEST(DenseTensorLayoutTest, UnsqueezeRejectsOutOfRange) {
  const std::array<Layout::Dim, 1> shape{3};
  Layout layout = Layout::contiguous(shape);

  ExpectError(Errc::OutOfRange, [&]() { (void)layout.unsqueeze(2); });
}

} // namespace
