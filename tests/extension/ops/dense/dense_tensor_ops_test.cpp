#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <system_error>

#include <orteaf/extension/ops/tensor_ops.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/init/library_init.h>
#include <orteaf/user/tensor/tensor.h>

namespace ops = ::orteaf::extension::ops;
namespace tensor = ::orteaf::user::tensor;
namespace init = ::orteaf::internal::init;

using DType = ::orteaf::internal::DType;
using Execution = ::orteaf::internal::execution::Execution;

class TensorOpsTest : public ::testing::Test {
protected:
  void SetUp() override { init::initialize(); }

  void TearDown() override { init::shutdown(); }
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
