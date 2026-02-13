#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/storage/registry/storage_types.h>
#include <orteaf/internal/tensor/manager/tensor_impl_manager.h>
#include <orteaf/internal/tensor/manager/tensor_impl_manager.inl>

namespace {

namespace cpu_api = orteaf::internal::execution::cpu::api;
namespace storage_reg = orteaf::internal::storage::registry;
using DenseTensorImpl = orteaf::extension::tensor::DenseTensorImpl;
using DenseTensorImplManager =
    orteaf::internal::tensor::TensorImplManager<DenseTensorImpl>;
using DType = orteaf::internal::DType;
using Execution = orteaf::internal::execution::Execution;
using StorageRegistry = storage_reg::RegisteredStorages;

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

auto createDense(DenseTensorImplManager &manager,
                 std::span<const std::int64_t> shape, DType dtype,
                 Execution execution, std::size_t alignment = 0) {
  return manager.create(makeDenseRequest(shape, dtype, execution, alignment));
}

class TensorImplManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config cpu_config{};
    cpu_api::CpuExecutionApi::configure(cpu_config);

    StorageRegistry::Config storage_config{};
    storage_registry_.configure(storage_config);

    DenseTensorImplManager::Config manager_config{};
    manager_.configure(manager_config, storage_registry_);
  }

  void TearDown() override {
    manager_.shutdown();
    storage_registry_.shutdown();
    cpu_api::CpuExecutionApi::shutdown();
  }

  StorageRegistry storage_registry_;
  DenseTensorImplManager manager_;
};

// =============================================================================
// TensorImplManager Creation Tests
// =============================================================================

TEST_F(TensorImplManagerTest, CreateTensorImpl) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = createDense(manager_, shape, DType::F32, Execution::Cpu);

  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->numel(), 12);
  EXPECT_EQ(lease->rank(), 2);
  EXPECT_EQ(lease->dtype(), DType::F32);
  EXPECT_EQ(lease->execution(), Execution::Cpu);
  EXPECT_TRUE(lease->isContiguous());
}

TEST_F(TensorImplManagerTest, CreateMultipleTensorImpls) {
  std::array<int64_t, 2> shape1{2, 3};
  std::array<int64_t, 3> shape2{2, 3, 4};

  auto lease1 = createDense(manager_, shape1, DType::F32, Execution::Cpu);
  auto lease2 = createDense(manager_, shape2, DType::F64, Execution::Cpu);

  ASSERT_TRUE(lease1);
  ASSERT_TRUE(lease2);
  EXPECT_EQ(lease1->numel(), 6);
  EXPECT_EQ(lease2->numel(), 24);
  EXPECT_EQ(lease1->dtype(), DType::F32);
  EXPECT_EQ(lease2->dtype(), DType::F64);
}

TEST_F(TensorImplManagerTest, CreateDifferentDTypes) {
  std::array<int64_t, 1> shape{10};

  auto f16 = createDense(manager_, shape, DType::F16, Execution::Cpu);
  auto f32 = createDense(manager_, shape, DType::F32, Execution::Cpu);
  auto f64 = createDense(manager_, shape, DType::F64, Execution::Cpu);
  auto i32 = createDense(manager_, shape, DType::I32, Execution::Cpu);

  EXPECT_EQ(f16->dtype(), DType::F16);
  EXPECT_EQ(f32->dtype(), DType::F32);
  EXPECT_EQ(f64->dtype(), DType::F64);
  EXPECT_EQ(i32->dtype(), DType::I32);
}

TEST_F(TensorImplManagerTest, CreateRejectsNegativeShapeDimension) {
  std::array<int64_t, 2> shape{-1, 4};
  EXPECT_THROW(createDense(manager_, shape, DType::F32, Execution::Cpu),
               std::system_error);
}

// =============================================================================
// View Operation Tests (Transpose)
// =============================================================================

TEST_F(TensorImplManagerTest, Transpose2D) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 2> perm{1, 0};
  auto transposed = manager_.transpose(original, perm);

  ASSERT_TRUE(transposed);
  auto t_shape = transposed->shape();
  EXPECT_EQ(t_shape[0], 4);
  EXPECT_EQ(t_shape[1], 3);
  EXPECT_EQ(transposed->numel(), 12);
  EXPECT_FALSE(transposed->isContiguous());
}

TEST_F(TensorImplManagerTest, Transpose3D) {
  std::array<int64_t, 3> shape{2, 3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 3> perm{2, 0, 1};
  auto transposed = manager_.transpose(original, perm);

  auto t_shape = transposed->shape();
  EXPECT_EQ(t_shape[0], 4);
  EXPECT_EQ(t_shape[1], 2);
  EXPECT_EQ(t_shape[2], 3);
}

// =============================================================================
// View Operation Tests (Reshape)
// =============================================================================

TEST_F(TensorImplManagerTest, ReshapeFlat) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 1> new_shape{12};
  auto reshaped = manager_.reshape(original, new_shape);

  ASSERT_TRUE(reshaped);
  EXPECT_EQ(reshaped->shape().size(), 1);
  EXPECT_EQ(reshaped->shape()[0], 12);
  EXPECT_TRUE(reshaped->isContiguous());
}

TEST_F(TensorImplManagerTest, ReshapeMultiDim) {
  std::array<int64_t, 2> shape{6, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 3> new_shape{2, 3, 4};
  auto reshaped = manager_.reshape(original, new_shape);

  EXPECT_EQ(reshaped->shape().size(), 3);
  EXPECT_EQ(reshaped->numel(), 24);
}

// =============================================================================
// View Operation Tests (Slice)
// =============================================================================

TEST_F(TensorImplManagerTest, Slice2D) {
  std::array<int64_t, 2> shape{6, 8};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<int64_t, 2> starts{1, 2};
  std::array<int64_t, 2> sizes{3, 4};
  auto sliced = manager_.slice(original, starts, sizes);

  ASSERT_TRUE(sliced);
  auto s_shape = sliced->shape();
  EXPECT_EQ(s_shape[0], 3);
  EXPECT_EQ(s_shape[1], 4);
  EXPECT_EQ(sliced->numel(), 12);
}

// =============================================================================
// View Operation Tests (Squeeze/Unsqueeze)
// =============================================================================

TEST_F(TensorImplManagerTest, Squeeze) {
  std::array<int64_t, 4> shape{1, 3, 1, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  auto squeezed = manager_.squeeze(original);

  EXPECT_EQ(squeezed->shape().size(), 2);
  EXPECT_EQ(squeezed->shape()[0], 3);
  EXPECT_EQ(squeezed->shape()[1], 4);
}

TEST_F(TensorImplManagerTest, SqueezeNoChange) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  auto squeezed = manager_.squeeze(original);

  EXPECT_EQ(squeezed->shape().size(), 2);
}

TEST_F(TensorImplManagerTest, Unsqueeze) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  auto unsqueezed = manager_.unsqueeze(original, 1);

  EXPECT_EQ(unsqueezed->shape().size(), 3);
  EXPECT_EQ(unsqueezed->shape()[0], 3);
  EXPECT_EQ(unsqueezed->shape()[1], 1);
  EXPECT_EQ(unsqueezed->shape()[2], 4);
}

TEST_F(TensorImplManagerTest, UnsqueezeAtEnd) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  auto unsqueezed = manager_.unsqueeze(original, 2);

  EXPECT_EQ(unsqueezed->shape().size(), 3);
  EXPECT_EQ(unsqueezed->shape()[2], 1);
}

// =============================================================================
// Storage Sharing Tests
// =============================================================================

TEST_F(TensorImplManagerTest, ViewsShareStorage) {
  std::array<int64_t, 2> shape{3, 4};
  auto original = createDense(manager_, shape, DType::F32, Execution::Cpu);

  std::array<std::size_t, 2> perm{1, 0};
  auto transposed = manager_.transpose(original, perm);

  // Both should have the same storage size (same underlying buffer)
  EXPECT_EQ(original->storageSizeInBytes(), transposed->storageSizeInBytes());
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TensorImplManagerTest, LeaseRelease) {
  std::array<int64_t, 2> shape{3, 4};

  {
    auto lease = createDense(manager_, shape, DType::F32, Execution::Cpu);
    EXPECT_TRUE(lease);
  }
  // Lease goes out of scope and should be released

  // Create another to verify pool is reusable
  auto lease2 = createDense(manager_, shape, DType::F32, Execution::Cpu);
  EXPECT_TRUE(lease2);
}

TEST_F(TensorImplManagerTest, IsConfigured) {
  EXPECT_TRUE(manager_.isConfigured());
}

TEST(TensorImplManagerRequestTest, CreateFailsWhenCpuExecutionIsUnconfigured) {
  StorageRegistry storage_registry;
  StorageRegistry::Config storage_config{};
  storage_registry.configure(storage_config);

  DenseTensorImplManager manager;
  DenseTensorImplManager::Config manager_config{};
  manager.configure(manager_config, storage_registry);

  std::array<int64_t, 2> shape{3, 4};
  EXPECT_THROW(createDense(manager, shape, DType::F32, Execution::Cpu),
               std::system_error);

  manager.shutdown();
  storage_registry.shutdown();
}

} // namespace
