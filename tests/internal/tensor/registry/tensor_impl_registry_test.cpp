#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include <orteaf/extension/tensor/registry/tensor_impl_types.h>
#include <orteaf/internal/init/library_init.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace {

namespace init = orteaf::internal::init;
namespace storage_reg = orteaf::internal::storage::registry;
namespace registry = orteaf::internal::tensor::registry;
using DenseTensorImpl = orteaf::extension::tensor::DenseTensorImpl;
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

class TensorImplRegistryTest : public ::testing::Test {
protected:
  void SetUp() override {
    init::LibraryConfig config{};
    config.register_kernels = false;
    init::initialize(config);

    StorageRegistry::Config storage_config{};
    storage_registry_.configure(storage_config);

    registry::RegisteredImpls::Config registry_config{};
    registry_.configure(registry_config, storage_registry_);
  }

  void TearDown() override {
    registry_.shutdown();
    storage_registry_.shutdown();
    init::shutdown();
  }

  StorageRegistry storage_registry_;
  registry::RegisteredImpls registry_;
};

// =============================================================================
// Registry Configuration Tests
// =============================================================================

TEST_F(TensorImplRegistryTest, IsConfigured) {
  EXPECT_TRUE(registry_.isConfigured());
}

TEST_F(TensorImplRegistryTest, GetDenseManager) {
  auto &manager = registry_.get<DenseTensorImpl>();
  EXPECT_TRUE(manager.isConfigured());
}

// =============================================================================
// LeaseVariant Tests
// =============================================================================

TEST_F(TensorImplRegistryTest, LeaseVariantHoldsMonostate) {
  registry::RegisteredImpls::LeaseVariant variant;
  EXPECT_TRUE(std::holds_alternative<std::monostate>(variant));
}

TEST_F(TensorImplRegistryTest, LeaseVariantHoldsDenseLease) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = registry_.get<DenseTensorImpl>().create(
      makeDenseRequest(shape, DType::F32, Execution::Cpu));

  registry::RegisteredImpls::LeaseVariant variant = lease;
  EXPECT_FALSE(std::holds_alternative<std::monostate>(variant));
}

// =============================================================================
// Dispatch Tests
// =============================================================================

TEST_F(TensorImplRegistryTest, DispatchToDenseManager) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = registry_.get<DenseTensorImpl>().create(
      makeDenseRequest(shape, DType::F32, Execution::Cpu));

  // Verify dispatch correctly identifies the impl type
  bool dispatched_correctly = false;

  registry::RegisteredImpls::dispatch(
      lease, [&]<typename Impl>(const auto & /*l*/) {
        if constexpr (std::is_same_v<Impl, DenseTensorImpl>) {
          dispatched_correctly = true;
        }
        return 0;
      });

  EXPECT_TRUE(dispatched_correctly);
}

// =============================================================================
// Traits Tests
// =============================================================================

TEST_F(TensorImplRegistryTest, DenseTraitsName) {
  EXPECT_STREQ(registry::TensorImplTraits<DenseTensorImpl>::name, "dense");
}

TEST_F(TensorImplRegistryTest, DenseTraitsLeaseType) {
  using Manager = orteaf::internal::tensor::TensorImplManager<DenseTensorImpl>;
  using Lease = typename Manager::TensorImplLease;

  // Verify types are correctly defined
  static_assert(std::is_same_v<Lease, typename Manager::TensorImplLease>);
}

} // namespace
