#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"
#include "orteaf/internal/storage/cpu/cpu_storage.h"

#include <gtest/gtest.h>

#include <memory>

namespace cpu_storage = orteaf::internal::storage::cpu;
namespace cpu = orteaf::internal::execution::cpu;
namespace cpu_manager = orteaf::internal::execution::cpu::manager;
namespace cpu_platform = orteaf::internal::execution::cpu::platform;
namespace cpu_api = orteaf::internal::execution::cpu::api;

class CpuStorageBuilderTest : public ::testing::Test {
protected:
  void SetUp() override {
    slow_ops_ = std::make_unique<cpu_platform::CpuSlowOpsImpl>();
    manager_ = std::make_unique<cpu_manager::CpuDeviceManager>();

    cpu_manager::CpuDeviceManager::Config config{};
    manager_->configureForTest(config, slow_ops_.get());
    device_lease_ = manager_->acquire(cpu::CpuDeviceHandle{0});
  }

  void TearDown() override {
    device_lease_.release();
    manager_->shutdown();
    manager_.reset();
    slow_ops_.reset();
  }

  std::unique_ptr<cpu_platform::CpuSlowOpsImpl> slow_ops_;
  std::unique_ptr<cpu_manager::CpuDeviceManager> manager_;
  cpu_manager::CpuDeviceManager::DeviceLease device_lease_{};
};

// ============================================================
// Builder tests
// ============================================================

TEST_F(CpuStorageBuilderTest, BuilderCreatesValidStorage) {
  auto storage = cpu_storage::CpuStorage::builder()
                     .withDeviceLease(device_lease_)
                     .withNumElements(1024)
                     .build();

  // Storage should be constructed (we can't check internal state easily,
  // but at least it shouldn't throw)
  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, BuilderWithAlignment) {
  auto storage = cpu_storage::CpuStorage::builder()
                     .withDeviceLease(device_lease_)
                     .withNumElements(512)
                     .withAlignment(64)
                     .build();

  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, BuilderWithLayout) {
  cpu_storage::CpuStorageLayout layout{};
  auto storage = cpu_storage::CpuStorage::builder()
                     .withDeviceLease(device_lease_)
                     .withNumElements(256)
                     .withLayout(layout)
                     .build();

  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, BuilderChainAllOptions) {
  cpu_storage::CpuStorageLayout layout{};
  auto storage = cpu_storage::CpuStorage::builder()
                     .withDeviceLease(device_lease_)
                     .withNumElements(1024)
                     .withAlignment(32)
                     .withLayout(layout)
                     .build();

  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, BuilderStaticMethodReturnsBuilder) {
  auto builder = cpu_storage::CpuStorage::builder();
  // Should be able to chain methods
  builder.withDeviceLease(device_lease_).withNumElements(128);
  auto storage = builder.build();

  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, DefaultConstructedStorageIsValid) {
  cpu_storage::CpuStorage storage;
  // Default constructed should be valid (just empty)
  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, StorageIsMoveConstructible) {
  auto storage1 = cpu_storage::CpuStorage::builder()
                      .withDeviceLease(device_lease_)
                      .withNumElements(1024)
                      .build();

  auto storage2 = std::move(storage1);
  SUCCEED();
}

TEST_F(CpuStorageBuilderTest, StorageIsMoveAssignable) {
  auto storage1 = cpu_storage::CpuStorage::builder()
                      .withDeviceLease(device_lease_)
                      .withNumElements(1024)
                      .build();

  cpu_storage::CpuStorage storage2;
  storage2 = std::move(storage1);
  SUCCEED();
}

TEST(CpuStorageExecutionApiTest, BuilderWithDeviceHandleUsesExecutionApi) {
  cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
  cpu_api::CpuExecutionApi::configure(config);
  {
    auto storage = cpu_storage::CpuStorage::builder()
                       .withDeviceHandle(cpu::CpuDeviceHandle{0})
                       .withNumElements(128)
                       .build();
    SUCCEED();
  }
  cpu_api::CpuExecutionApi::shutdown();
}
