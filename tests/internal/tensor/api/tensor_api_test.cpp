#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/tensor/api/tensor_api.h>

namespace {

namespace tensor_api = orteaf::internal::tensor::api;
namespace cpu_api = orteaf::internal::execution::cpu::api;
using DenseTensorImpl = orteaf::extension::tensor::DenseTensorImpl;
using DType = orteaf::internal::DType;
using Execution = orteaf::internal::execution::Execution;

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

auto createDense(std::span<const std::int64_t> shape, DType dtype,
                 Execution execution, std::size_t alignment = 0) {
  return tensor_api::TensorApi::create<DenseTensorImpl>(
      makeDenseRequest(shape, dtype, execution, alignment));
}

class TensorApiInternalTest : public ::testing::Test {
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
// Configuration Tests
// =============================================================================

TEST_F(TensorApiInternalTest, IsConfigured) {
  EXPECT_TRUE(tensor_api::TensorApi::isConfigured());
}

TEST_F(TensorApiInternalTest, DoubleConfigureThrows) {
  // Already configured in SetUp
  tensor_api::TensorApi::Config config{};
  EXPECT_THROW(tensor_api::TensorApi::configure(config), std::system_error);
}

// =============================================================================
// CreateTemplate Tests
// =============================================================================

TEST_F(TensorApiInternalTest, CreateDenseViaTemplate) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->numel(), 12);
  EXPECT_EQ(lease->dtype(), DType::F32);
}

TEST_F(TensorApiInternalTest, CreateDifferentShapes) {
  std::array<int64_t, 1> shape1{10};
  std::array<int64_t, 4> shape2{2, 3, 4, 5};

  auto lease1 = createDense(
      shape1, DType::F32, Execution::Cpu);
  auto lease2 = createDense(
      shape2, DType::F32, Execution::Cpu);

  EXPECT_EQ(lease1->rank(), 1);
  EXPECT_EQ(lease2->rank(), 4);
  EXPECT_EQ(lease2->numel(), 120);
}

// =============================================================================
// Auto-Dispatch Transpose Tests
// =============================================================================

TEST_F(TensorApiInternalTest, AutoDispatchTranspose) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  std::array<std::size_t, 2> perm{1, 0};

  auto result = tensor_api::TensorApi::transpose(variant, perm);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(result));
}

TEST_F(TensorApiInternalTest, TransposePreservesType) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  std::array<std::size_t, 2> perm{1, 0};

  auto result = tensor_api::TensorApi::transpose(variant, perm);

  // Result should still be a dense tensor
  using DenseLease = typename orteaf::internal::tensor::TensorImplManager<
      DenseTensorImpl>::TensorImplLease;
  EXPECT_TRUE(std::holds_alternative<DenseLease>(result));
}

// =============================================================================
// Auto-Dispatch Reshape Tests
// =============================================================================

TEST_F(TensorApiInternalTest, AutoDispatchReshape) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  std::array<int64_t, 1> new_shape{12};

  auto result = tensor_api::TensorApi::reshape(variant, new_shape);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(result));
}

// =============================================================================
// Auto-Dispatch Slice Tests
// =============================================================================

TEST_F(TensorApiInternalTest, AutoDispatchSlice) {
  std::array<int64_t, 2> shape{6, 8};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  std::array<int64_t, 2> starts{1, 2};
  std::array<int64_t, 2> sizes{3, 4};

  auto result = tensor_api::TensorApi::slice(variant, starts, sizes);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(result));
}

// =============================================================================
// Auto-Dispatch Squeeze/Unsqueeze Tests
// =============================================================================

TEST_F(TensorApiInternalTest, AutoDispatchSqueeze) {
  std::array<int64_t, 3> shape{1, 4, 1};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  auto result = tensor_api::TensorApi::squeeze(variant);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(result));
}

TEST_F(TensorApiInternalTest, AutoDispatchUnsqueeze) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;
  auto result = tensor_api::TensorApi::unsqueeze(variant, 0);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(result));
}

// =============================================================================
// Invalid State Tests
// =============================================================================

TEST_F(TensorApiInternalTest, TransposeInvalidThrows) {
  tensor_api::TensorApi::LeaseVariant invalid_variant;
  std::array<std::size_t, 2> perm{1, 0};

  EXPECT_THROW(tensor_api::TensorApi::transpose(invalid_variant, perm),
               std::system_error);
}

TEST_F(TensorApiInternalTest, ReshapeInvalidThrows) {
  tensor_api::TensorApi::LeaseVariant invalid_variant;
  std::array<int64_t, 1> new_shape{12};

  EXPECT_THROW(tensor_api::TensorApi::reshape(invalid_variant, new_shape),
               std::system_error);
}

TEST_F(TensorApiInternalTest, SqueezeInvalidThrows) {
  tensor_api::TensorApi::LeaseVariant invalid_variant;
  EXPECT_THROW(tensor_api::TensorApi::squeeze(invalid_variant),
               std::system_error);
}

// =============================================================================
// Chained Operations Tests
// =============================================================================

TEST_F(TensorApiInternalTest, ChainedAutoDispatch) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(shape, DType::F32,
                                                              Execution::Cpu);

  tensor_api::TensorApi::LeaseVariant variant = lease;

  // Chain: transpose -> unsqueeze
  std::array<std::size_t, 2> perm{1, 0};
  auto transposed = tensor_api::TensorApi::transpose(variant, perm);
  auto unsqueezed = tensor_api::TensorApi::unsqueeze(transposed, 0);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(unsqueezed));
}

} // namespace
