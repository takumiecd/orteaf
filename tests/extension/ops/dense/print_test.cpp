#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
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

class PrintOpTest : public ::testing::Test {
protected:
  void SetUp() override { init::initialize(); }

  void TearDown() override { init::shutdown(); }
};

TEST_F(PrintOpTest, PrintsDenseTensorToStdout) {
  std::array<std::int64_t, 2> shape{2, 2};
  auto t = makeDense(shape, DType::F32, Execution::Cpu);

  float *data = getCpuBuffer(t);
  ASSERT_NE(data, nullptr);
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  std::ostringstream oss;
  auto *old = std::cout.rdbuf(oss.rdbuf());
  ops::TensorOps::print(t);
  std::cout.rdbuf(old);

  const auto output = oss.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_FALSE(output.empty());
}

TEST_F(PrintOpTest, ThrowsOnNonContiguousView) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = makeDense(shape, DType::F32, Execution::Cpu);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto view = base.slice(starts, sizes);

  EXPECT_ANY_THROW(ops::TensorOps::print(view));
}
