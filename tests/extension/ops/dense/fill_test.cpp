#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/init/library_init.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/user/tensor/tensor.h>

namespace ops = ::orteaf::extension::ops;
namespace tensor = ::orteaf::user::tensor;
namespace init = ::orteaf::internal::init;

using DType = ::orteaf::internal::DType;
using Execution = ::orteaf::internal::execution::Execution;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
using CpuStorageLease = ::orteaf::internal::storage::CpuStorageLease;

namespace {

tensor::Tensor makeDense(std::span<const std::int64_t> shape, DType dtype,
                         Execution execution, std::size_t alignment = 0) {
  return tensor::Tensor::denseBuilder()
      .withShape(shape)
      .withDType(dtype)
      .withExecution(execution)
      .withAlignment(alignment)
      .build();
}

float *getCpuBuffer(tensor::Tensor &t) {
  auto *lease = t.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease)) {
    return nullptr;
  }
  auto *impl = lease->operator->();
  if (impl == nullptr) {
    return nullptr;
  }
  auto *cpu_lease = impl->storageLease().tryAs<CpuStorageLease>();
  if (!cpu_lease || !(*cpu_lease)) {
    return nullptr;
  }
  auto *cpu_storage = cpu_lease->operator->();
  if (cpu_storage == nullptr) {
    return nullptr;
  }
  return static_cast<float *>(cpu_storage->buffer());
}

}  // namespace

class FillOpTest : public ::testing::Test {
protected:
  void SetUp() override { init::initialize(); }

  void TearDown() override { init::shutdown(); }
};

TEST_F(FillOpTest, FillsDenseTensor) {
  std::array<std::int64_t, 2> shape{2, 3};
  auto t = makeDense(shape, DType::F32, Execution::Cpu);

  ops::TensorOps::fill(t, 1.5);

  float *data = getCpuBuffer(t);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(t.numel()); ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.5f);
  }
}

TEST_F(FillOpTest, FillsStridedSliceView) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = makeDense(shape, DType::F32, Execution::Cpu);

  float *data = getCpuBuffer(base);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(base.numel()); ++i) {
    data[i] = static_cast<float>(i);
  }

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto view = base.slice(starts, sizes);
  ASSERT_TRUE(view.valid());

  ops::TensorOps::fill(view, -2.0);

  for (std::size_t i = 0; i < static_cast<std::size_t>(base.numel()); ++i) {
    const bool should_fill =
        (i == 4 || i == 5 || i == 6 || i == 8 || i == 9 || i == 10);
    if (should_fill) {
      EXPECT_FLOAT_EQ(data[i], -2.0f);
    } else {
      EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
  }
}
