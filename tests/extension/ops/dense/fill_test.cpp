#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/tensor/api/tensor_api.h>
#include <orteaf/user/tensor/tensor.h>

namespace ops = ::orteaf::extension::ops;
namespace tensor = ::orteaf::user::tensor;
namespace tensor_api = ::orteaf::internal::tensor::api;
namespace cpu_api = ::orteaf::internal::execution::cpu::api;
namespace kernel_registry = ::orteaf::internal::kernel::registry;
namespace kernel_api = ::orteaf::internal::kernel::api;
namespace cpu_context = ::orteaf::internal::execution_context::cpu;

using DType = ::orteaf::internal::DType;
using Execution = ::orteaf::internal::execution::Execution;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
using CpuStorageLease = ::orteaf::internal::storage::CpuStorageLease;

namespace {

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
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config cpu_config{};
    cpu_api::CpuExecutionApi::configure(cpu_config);

    tensor_api::TensorApi::Config tensor_config{};
    tensor_api::TensorApi::configure(tensor_config);

    kernel_registry::registerAllKernels();
  }

  void TearDown() override {
    cpu_context::reset();
    kernel_api::KernelRegistryApi::clear();
    tensor_api::TensorApi::shutdown();
    cpu_api::CpuExecutionApi::shutdown();
  }
};

TEST_F(FillOpTest, FillsDenseTensor) {
  std::array<std::int64_t, 2> shape{2, 3};
  auto t = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  ops::TensorOps::fill(t, 1.5);

  float *data = getCpuBuffer(t);
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < static_cast<std::size_t>(t.numel()); ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.5f);
  }
}

TEST_F(FillOpTest, FillsStridedSliceView) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

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
