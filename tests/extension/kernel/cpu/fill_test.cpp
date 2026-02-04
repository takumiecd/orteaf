#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/dtype/float16.h>

namespace orteaf::extension::kernel::cpu {
void fillTensor(void *data, std::size_t count, ::orteaf::internal::DType dtype,
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

}  // namespace
