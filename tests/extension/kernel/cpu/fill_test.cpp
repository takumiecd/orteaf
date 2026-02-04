#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/dtype/float16.h>

namespace orteaf::extension::kernel::cpu {
void fillTensor(void *data, std::size_t count, ::orteaf::internal::DType dtype,
                double value);
void fillTensorStrided(void *data, std::span<const std::int64_t> shape,
                       std::span<const std::int64_t> strides,
                       std::int64_t offset, ::orteaf::internal::DType dtype,
                       double value);
}

namespace {

TEST(CpuFillTest, FillsFloat32) {
  std::array<float, 4> data{};

  orteaf::extension::kernel::cpu::fillTensor(
      data.data(), data.size(), orteaf::internal::DType::F32, 1.25);

  for (float v : data) {
    EXPECT_FLOAT_EQ(v, 1.25f);
  }
}

TEST(CpuFillTest, FillsInt32WithCast) {
  std::array<std::int32_t, 3> data{};

  orteaf::extension::kernel::cpu::fillTensor(
      data.data(), data.size(), orteaf::internal::DType::I32, 7.9);

  for (std::int32_t v : data) {
    EXPECT_EQ(v, 7);
  }
}

TEST(CpuFillTest, FillsBool) {
  std::array<bool, 2> data{};

  orteaf::extension::kernel::cpu::fillTensor(
      data.data(), data.size(), orteaf::internal::DType::Bool, 0.0);
  EXPECT_FALSE(data[0]);
  EXPECT_FALSE(data[1]);

  orteaf::extension::kernel::cpu::fillTensor(
      data.data(), data.size(), orteaf::internal::DType::Bool, 2.0);
  EXPECT_TRUE(data[0]);
  EXPECT_TRUE(data[1]);
}

TEST(CpuFillTest, FillsFloat16) {
  std::array<::orteaf::internal::Float16, 2> data{};

  orteaf::extension::kernel::cpu::fillTensor(
      data.data(), data.size(), orteaf::internal::DType::F16, 1.5);

  const ::orteaf::internal::Float16 expected(1.5f);
  EXPECT_EQ(data[0], expected);
  EXPECT_EQ(data[1], expected);
}

TEST(CpuFillTest, ThrowsOnNullBuffer) {
  EXPECT_ANY_THROW(orteaf::extension::kernel::cpu::fillTensor(
      nullptr, 1, orteaf::internal::DType::F32, 0.0));
}

TEST(CpuFillTest, FillsStridedView) {
  std::vector<float> data(12);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(100 + i);
  }

  const std::array<std::int64_t, 2> shape{{2, 3}};
  const std::array<std::int64_t, 2> strides{{5, 1}};
  const std::int64_t offset = 2;

  orteaf::extension::kernel::cpu::fillTensorStrided(
      data.data(), std::span<const std::int64_t>(shape.data(), shape.size()),
      std::span<const std::int64_t>(strides.data(), strides.size()), offset,
      orteaf::internal::DType::F32, 3.5);

  for (std::size_t i = 0; i < data.size(); ++i) {
    const bool should_fill =
        (i == 2 || i == 3 || i == 4 || i == 7 || i == 8 || i == 9);
    if (should_fill) {
      EXPECT_FLOAT_EQ(data[i], 3.5f);
    } else {
      EXPECT_FLOAT_EQ(data[i], static_cast<float>(100 + i));
    }
  }
}

TEST(CpuFillTest, FillsOffsetView1D) {
  std::vector<float> data(10);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(200 + i);
  }

  const std::array<std::int64_t, 1> shape{{4}};
  const std::array<std::int64_t, 1> strides{{1}};
  const std::int64_t offset = 3;

  orteaf::extension::kernel::cpu::fillTensorStrided(
      data.data(), std::span<const std::int64_t>(shape.data(), shape.size()),
      std::span<const std::int64_t>(strides.data(), strides.size()), offset,
      orteaf::internal::DType::F32, -1.0);

  for (std::size_t i = 0; i < data.size(); ++i) {
    const bool should_fill = (i >= 3 && i <= 6);
    if (should_fill) {
      EXPECT_FLOAT_EQ(data[i], -1.0f);
    } else {
      EXPECT_FLOAT_EQ(data[i], static_cast<float>(200 + i));
    }
  }
}

}  // namespace
