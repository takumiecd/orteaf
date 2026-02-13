#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include <orteaf/internal/init/library_init.h>
#include <orteaf/internal/tensor/api/tensor_api.h>
#include <orteaf/user/tensor/tensor.h>

namespace tensor_api = orteaf::internal::tensor::api;
namespace tensor = orteaf::user::tensor;
namespace init = orteaf::internal::init;
using DType = orteaf::internal::DType;
using Execution = orteaf::internal::execution::Execution;
using DenseTensorImpl = orteaf::extension::tensor::DenseTensorImpl;

namespace {

DenseTensorImpl::CreateRequest makeDenseRequest(std::span<const std::int64_t> shape,
                                                DType dtype,
                                                Execution execution,
                                                std::size_t alignment = 0) {
  DenseTensorImpl::CreateRequest request{};
  request.shape.assign(shape.begin(), shape.end());
  request.dtype = dtype;
  request.execution = execution;
  request.alignment = alignment;
  return request;
}

tensor::Tensor makeDense(std::span<const std::int64_t> shape, DType dtype,
                         Execution execution, std::size_t alignment = 0) {
  return tensor::Tensor::denseBuilder()
      .withShape(shape)
      .withDType(dtype)
      .withExecution(execution)
      .withAlignment(alignment)
      .build();
}

} // namespace

class TensorApiTest : public ::testing::Test {
protected:
  void SetUp() override {
    init::LibraryConfig config{};
    config.register_kernels = false;
    init::initialize(config);
  }

  void TearDown() override { init::shutdown(); }
};

// =============================================================================
// TensorApi Tests (using template create)
// =============================================================================

TEST_F(TensorApiTest, CreateDenseTensor) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = tensor_api::TensorApi::create<DenseTensorImpl>(
      makeDenseRequest(shape, DType::F32, Execution::Cpu));

  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->numel(), 12);
  EXPECT_EQ(lease->rank(), 2);
}

TEST_F(TensorApiTest, AutoDispatchTranspose) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = tensor_api::TensorApi::create<DenseTensorImpl>(
      makeDenseRequest(shape, DType::F32, Execution::Cpu));

  // Auto-dispatch via LeaseVariant
  tensor_api::TensorApi::LeaseVariant variant = original;
  std::array<std::size_t, 2> perm{1, 0};
  auto transposed = tensor_api::TensorApi::transpose(variant, perm);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(transposed));
}

// =============================================================================
// Tensor Class Tests
// =============================================================================

TEST_F(TensorApiTest, TensorDenseFactory) {
  std::array<int64_t, 2> shape{3, 4};
  auto t = makeDense(shape, DType::F32, Execution::Cpu);

  EXPECT_TRUE(t.valid());
  EXPECT_TRUE(t.is<DenseTensorImpl>());
  EXPECT_EQ(t.numel(), 12);
  EXPECT_EQ(t.rank(), 2);
}

TEST_F(TensorApiTest, DenseBuilderRequiresShape) {
  EXPECT_THROW(tensor::Tensor::denseBuilder()
                   .withDType(DType::F32)
                   .withExecution(Execution::Cpu)
                   .build(),
               std::system_error);
}

TEST_F(TensorApiTest, DenseBuilderRequiresExecution) {
  std::array<int64_t, 2> shape{3, 4};
  EXPECT_THROW(
      tensor::Tensor::denseBuilder().withShape(shape).withDType(DType::F32).build(),
      std::system_error);
}

TEST_F(TensorApiTest, TensorTranspose) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 2> perm{1, 0};
  auto b = a.transpose(perm);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape[0], 4);
  EXPECT_EQ(b_shape[1], 3);
}

TEST_F(TensorApiTest, TensorReshape) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 3> new_shape{2, 2, 3};
  auto b = a.reshape(new_shape);

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.numel(), 12);
}

TEST_F(TensorApiTest, TensorSlice) {
  std::array<int64_t, 2> shape{6, 6};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> starts{2, 2};
  std::array<int64_t, 2> sizes{3, 3};
  auto b = a.slice(starts, sizes);

  EXPECT_TRUE(b.valid());
  auto b_shape = b.shape();
  EXPECT_EQ(b_shape[0], 3);
  EXPECT_EQ(b_shape[1], 3);
}

TEST_F(TensorApiTest, TensorSqueeze) {
  std::array<int64_t, 4> shape{1, 3, 1, 4};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  auto b = a.squeeze();

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.shape().size(), 2);
}

TEST_F(TensorApiTest, TensorUnsqueeze) {
  std::array<int64_t, 2> shape{3, 4};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  auto b = a.unsqueeze(1);

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.shape().size(), 3);
  EXPECT_EQ(b.shape()[1], 1);
}

TEST_F(TensorApiTest, TensorChainOperations) {
  std::array<int64_t, 3> shape{2, 3, 4};
  auto a = makeDense(shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> reshape_to{6, 4};
  std::array<std::size_t, 2> perm{1, 0};

  auto b = a.reshape(reshape_to).transpose(perm).unsqueeze(0);

  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.shape().size(), 3);
  EXPECT_EQ(b.shape()[0], 1);
}

TEST_F(TensorApiTest, InvalidTensorThrows) {
  tensor::Tensor invalid_tensor;

  EXPECT_FALSE(invalid_tensor.valid());
  EXPECT_THROW(invalid_tensor.dtype(), std::system_error);
}
