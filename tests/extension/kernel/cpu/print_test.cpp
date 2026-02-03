#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>

#include <orteaf/extension/kernel/cpu/print.h>
#include <orteaf/internal/base/inline_vector.h>
#include <orteaf/internal/dtype/dtype.h>

namespace {

TEST(CpuPrintTest, Prints2DInt32) {
  std::array<std::int32_t, 6> data{{1, 2, 3, 4, 5, 6}};
  std::array<std::int64_t, 2> shape{{2, 3}};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::I32, os);

  EXPECT_EQ(os.str(), "[\n  [1, 2, 3],\n  [4, 5, 6]\n]");
}

TEST(CpuPrintTest, Prints1DInt64) {
  std::array<std::int64_t, 3> data{{7, 8, 9}};
  std::array<std::int64_t, 1> shape{{3}};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::I64, os);

  EXPECT_EQ(os.str(), "[7, 8, 9]");
}

TEST(CpuPrintTest, PrintsScalar) {
  std::array<std::int32_t, 1> data{{42}};
  std::array<std::int64_t, 0> shape{};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::I32, os);

  EXPECT_EQ(os.str(), "42");
}

TEST(CpuPrintTest, PrintsBoolInlineVectorShape) {
  std::array<bool, 2> data{{true, false}};
  ::orteaf::internal::base::InlineVector<std::int64_t, 4> shape{};
  shape.size = 2;
  shape.data[0] = 1;
  shape.data[1] = 2;

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(shape, data.data(),
                                              orteaf::internal::DType::Bool,
                                              os);

  EXPECT_EQ(os.str(), "[\n  [true, false]\n]");
}

TEST(CpuPrintTest, Prints2DFloat32) {
  std::array<float, 4> data{{1.5f, -2.25f, 3.0f, 0.0f}};
  std::array<std::int64_t, 2> shape{{2, 2}};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::F32, os);

  EXPECT_EQ(os.str(), "[\n  [1.5, -2.25],\n  [3, 0]\n]");
}

TEST(CpuPrintTest, Prints1DFloat64) {
  std::array<double, 3> data{{0.125, -2.5, 100.0}};
  std::array<std::int64_t, 1> shape{{3}};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::F64, os);

  EXPECT_EQ(os.str(), "[0.125, -2.5, 100]");
}

TEST(CpuPrintTest, RejectsStrideOverflow) {
  std::array<std::int32_t, 1> data{{0}};
  const std::uint64_t max_size =
      std::numeric_limits<std::size_t>::max();
  const std::uint64_t max_i64 =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const std::uint64_t half_plus_one = max_size / 2 + 1;
  const std::uint64_t capped =
      half_plus_one > max_i64 ? max_i64 : half_plus_one;
  const std::int64_t big = static_cast<std::int64_t>(capped);
  std::array<std::int64_t, 3> shape{{1, big, big}};

  std::ostringstream os;
  orteaf::extension::kernel::cpu::printTensor(
      std::span<const std::int64_t>(shape.data(), shape.size()), data.data(),
      orteaf::internal::DType::I32, os);

  EXPECT_EQ(os.str(), "<invalid shape>");
}

}  // namespace
