#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>

#include <orteaf/extension/ops/print.h>
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

class PrintOpTest : public ::testing::Test {
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

TEST_F(PrintOpTest, PrintsDenseTensorToStdout) {
  std::array<std::int64_t, 2> shape{2, 2};
  auto t = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  float *data = getCpuBuffer(t);
  ASSERT_NE(data, nullptr);
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  std::ostringstream oss;
  auto *old = std::cout.rdbuf(oss.rdbuf());
  ops::print(t);
  std::cout.rdbuf(old);

  const auto output = oss.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_FALSE(output.empty());
}

TEST_F(PrintOpTest, ThrowsOnNonContiguousView) {
  std::array<std::int64_t, 2> shape{4, 4};
  auto base = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  std::array<std::int64_t, 2> starts{1, 0};
  std::array<std::int64_t, 2> sizes{2, 3};
  auto view = base.slice(starts, sizes);

  EXPECT_ANY_THROW(ops::print(view));
}
