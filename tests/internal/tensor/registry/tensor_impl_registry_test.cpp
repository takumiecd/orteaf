#include <gtest/gtest.h>

#include <orteaf/extension/tensor/registry/tensor_impl_types.h>
#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/storage/manager/storage_manager.h>

namespace {

namespace cpu_api = orteaf::internal::execution::cpu::api;
namespace storage_mgr = orteaf::internal::storage::manager;
namespace registry = orteaf::internal::tensor::registry;
using DenseTensorImpl = orteaf::extension::tensor::DenseTensorImpl;
using DType = orteaf::internal::DType;
using Execution = orteaf::internal::execution::Execution;

class TensorImplRegistryTest : public ::testing::Test {
protected:
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config cpu_config{};
    cpu_api::CpuExecutionApi::configure(cpu_config);

    storage_mgr::StorageManager::Config storage_config{};
    storage_manager_.configure(storage_config);

    registry::RegisteredImpls::Config registry_config{};
    registry_.configure(registry_config, storage_manager_);
  }

  void TearDown() override {
    registry_.shutdown();
    storage_manager_.shutdown();
    cpu_api::CpuExecutionApi::shutdown();
  }

  storage_mgr::StorageManager storage_manager_;
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
  auto lease = registry_.get<DenseTensorImpl>().create(shape, DType::F32,
                                                       Execution::Cpu);

  registry::RegisteredImpls::LeaseVariant variant = lease;
  EXPECT_FALSE(std::holds_alternative<std::monostate>(variant));
}

// =============================================================================
// Dispatch Tests
// =============================================================================

TEST_F(TensorImplRegistryTest, DispatchToDenseManager) {
  std::array<int64_t, 2> shape{3, 4};
  auto lease = registry_.get<DenseTensorImpl>().create(shape, DType::F32,
                                                       Execution::Cpu);

  // Verify dispatch correctly identifies the impl type
  using Lease = registry::TensorImplTraits<DenseTensorImpl>::Lease;
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
  using Lease = registry::TensorImplTraits<DenseTensorImpl>::Lease;
  using Manager = registry::TensorImplTraits<DenseTensorImpl>::Manager;

  // Verify types are correctly defined
  static_assert(std::is_same_v<Lease, typename Manager::TensorImplLease>);
}

} // namespace
