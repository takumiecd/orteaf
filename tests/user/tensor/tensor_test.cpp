#include <gtest/gtest.h>

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/tensor/api/tensor_api.h>
#include <orteaf/user/tensor/tensor.h>

namespace tensor_api = orteaf::internal::tensor::api;
namespace tensor = orteaf::user::tensor;
namespace cpu_api = orteaf::internal::execution::cpu::api;
using DType = orteaf::internal::DType;
using Execution = orteaf::internal::execution::Execution;

class TensorApiTest : public ::testing::Test {
protected:
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config cpu_config{};
    cpu_api::CpuExecutionApi::configure(cpu_config);

    tensor_api::TensorApi::Config config{};
    tensor_api::TensorApi::configure(config);
  }

  void TearDown() override {
    tensor_api::TensorApi::shutdown();
    cpu_api::CpuExecutionApi::shutdown();
  }
};

// =============================================================================
// TensorApi Tests
// =============================================================================

TEST_F(TensorApiTest, CreateDenseTensor) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->numel(), 12);
  EXPECT_EQ(lease->rank(), 2);
  EXPECT_EQ(lease->dtype(), DType::F32);
  EXPECT_EQ(lease->execution(), Execution::Cpu);
}

TEST_F(TensorApiTest, TransposeTensor) {
  std::array<int64_t, 2> shape{3, 4};
  auto original =
      tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 2> perm{1, 0};
  auto transposed = tensor_api::TensorApi::transpose(original, perm);

  ASSERT_TRUE(transposed);
  auto t_shape = transposed->shape();
  EXPECT_EQ(t_shape.size(), 2);
  EXPECT_EQ(t_shape[0], 4);
  EXPECT_EQ(t_shape[1], 3);
  EXPECT_EQ(transposed->numel(), 12);
}

TEST_F(TensorApiTest, ReshapeTensor) {
  std::array<int64_t, 2> shape{3, 4};
  auto original =
      tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 1> new_shape{12};
  auto reshaped = tensor_api::TensorApi::reshape(original, new_shape);

  ASSERT_TRUE(reshaped);
  auto r_shape = reshaped->shape();
  EXPECT_EQ(r_shape.size(), 1);
  EXPECT_EQ(r_shape[0], 12);
}

TEST_F(TensorApiTest, SliceTensor) {
  std::array<int64_t, 2> shape{4, 4};
  auto original =
      tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> starts{1, 1};
  std::array<int64_t, 2> sizes{2, 2};
  auto sliced = tensor_api::TensorApi::slice(original, starts, sizes);

  ASSERT_TRUE(sliced);
  auto s_shape = sliced->shape();
  EXPECT_EQ(s_shape.size(), 2);
  EXPECT_EQ(s_shape[0], 2);
  EXPECT_EQ(s_shape[1], 2);
  EXPECT_EQ(sliced->numel(), 4);
}

TEST_F(TensorApiTest, SqueezeTensor) {
  std::array<int64_t, 3> shape{1, 3, 1};
  auto original =
      tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  auto squeezed = tensor_api::TensorApi::squeeze(original);

  ASSERT_TRUE(squeezed);
  auto s_shape = squeezed->shape();
  EXPECT_EQ(s_shape.size(), 1);
  EXPECT_EQ(s_shape[0], 3);
}

TEST_F(TensorApiTest, UnsqueezeTensor) {
  std::array<int64_t, 2> shape{3, 4};
  auto original =
      tensor_api::TensorApi::create(shape, DType::F32, Execution::Cpu);

  auto unsqueezed = tensor_api::TensorApi::unsqueeze(original, 0);

  ASSERT_TRUE(unsqueezed);
  auto u_shape = unsqueezed->shape();
  EXPECT_EQ(u_shape.size(), 3);
  EXPECT_EQ(u_shape[0], 1);
  EXPECT_EQ(u_shape[1], 3);
  EXPECT_EQ(u_shape[2], 4);
}

// =============================================================================
// Tensor Class Tests
// =============================================================================

TEST_F(TensorApiTest, TensorDenseFactory) {
  std::array<int64_t, 2> shape{3, 4};
  auto t = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  EXPECT_TRUE(t.valid());
  EXPECT_TRUE(t.isDense());
  EXPECT_EQ(t.numel(), 12);
  EXPECT_EQ(t.rank(), 2);
  EXPECT_EQ(t.dtype(), DType::F32);
  EXPECT_EQ(t.execution(), Execution::Cpu);
  EXPECT_TRUE(t.isContiguous());
}

TEST_F(TensorApiTest, TensorTranspose) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 2> perm{1, 0};
  auto b = a.transpose(perm);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape[0], 4);
  EXPECT_EQ(b_shape[1], 3);
  EXPECT_FALSE(b.isContiguous());
}

TEST_F(TensorApiTest, TensorReshape) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 3> new_shape{2, 2, 3};
  auto b = a.reshape(new_shape);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape.size(), 3);
  EXPECT_EQ(b_shape[0], 2);
  EXPECT_EQ(b_shape[1], 2);
  EXPECT_EQ(b_shape[2], 3);
  EXPECT_EQ(b.numel(), 12);
}

TEST_F(TensorApiTest, TensorSlice) {
  std::array<int64_t, 2> shape{6, 6};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> starts{2, 2};
  std::array<int64_t, 2> sizes{3, 3};
  auto b = a.slice(starts, sizes);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape[0], 3);
  EXPECT_EQ(b_shape[1], 3);
  EXPECT_EQ(b.numel(), 9);
}

TEST_F(TensorApiTest, TensorSqueeze) {
  std::array<int64_t, 4> shape{1, 3, 1, 4};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  auto b = a.squeeze();

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape.size(), 2);
  EXPECT_EQ(b_shape[0], 3);
  EXPECT_EQ(b_shape[1], 4);
}

TEST_F(TensorApiTest, TensorUnsqueeze) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  auto b = a.unsqueeze(1);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape.size(), 3);
  EXPECT_EQ(b_shape[0], 3);
  EXPECT_EQ(b_shape[1], 1);
  EXPECT_EQ(b_shape[2], 4);
}

TEST_F(TensorApiTest, TensorChainOperations) {
  std::array<int64_t, 3> shape{2, 3, 4};
  auto a = tensor::Tensor::dense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> reshape_to{6, 4};
  std::array<std::size_t, 2> perm{1, 0};

  auto b = a.reshape(reshape_to).transpose(perm).unsqueeze(0);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape.size(), 3);
  EXPECT_EQ(b_shape[0], 1);
  EXPECT_EQ(b_shape[1], 4);
  EXPECT_EQ(b_shape[2], 6);
}

TEST_F(TensorApiTest, InvalidTensorThrows) {
  tensor::Tensor invalid_tensor;

  EXPECT_FALSE(invalid_tensor.valid());
  EXPECT_THROW(invalid_tensor.dtype(), std::system_error);
  EXPECT_THROW(invalid_tensor.shape(), std::system_error);
}
