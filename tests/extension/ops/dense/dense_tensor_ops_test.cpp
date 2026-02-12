#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <system_error>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>
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

class TensorOpsTest : public ::testing::Test {
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

TEST_F(TensorOpsTest, InvalidTensorThrowsOnFill) {
  tensor::Tensor invalid;
  EXPECT_THROW(ops::TensorOps::fill(invalid, 1.0), std::system_error);
}

TEST_F(TensorOpsTest, NonDenseRoutingPendingNonDenseImpl) {
  GTEST_SKIP() << "Only DenseTensorImpl is registered in TensorImplRegistry; "
                  "non-dense routing will be validated when an additional "
                  "TensorImpl is available.";
}

TEST_F(TensorOpsTest, InvalidTensorThrowsOnPrint) {
  tensor::Tensor invalid;
  EXPECT_THROW(ops::TensorOps::print(invalid), std::system_error);
}

TEST_F(TensorOpsTest, PrintAcceptsConstTensor) {
  std::array<std::int64_t, 2> shape{2, 2};
  auto tensor_value = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  const tensor::Tensor &const_tensor = tensor_value;
  std::ostringstream oss;
  auto *old = std::cout.rdbuf(oss.rdbuf());
  EXPECT_NO_THROW(ops::TensorOps::print(const_tensor));
  std::cout.rdbuf(old);

  EXPECT_FALSE(oss.str().empty());
}
