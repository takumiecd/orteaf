#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cpu_detect.h"
#include "orteaf/internal/execution/cpu/manager/cpu_runtime_manager.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

namespace base = orteaf::internal::base;
namespace cpu_rt = orteaf::internal::execution::cpu::manager;
namespace cpu_platform = orteaf::internal::execution::cpu::platform;
namespace architecture = orteaf::internal::architecture;
using ::testing::NiceMock;
using ::testing::Return;

namespace {

/**
 * @brief Mock implementation of CpuSlowOps for testing.
 */
class CpuSlowOpsMock : public cpu_platform::CpuSlowOps {
public:
  MOCK_METHOD(int, getDeviceCount, (), (override));
  MOCK_METHOD(architecture::Architecture, detectArchitecture,
              (base::DeviceHandle device_id), (override));
  MOCK_METHOD(void *, allocBuffer, (std::size_t size, std::size_t alignment),
              (override));
  MOCK_METHOD(void, deallocBuffer, (void *ptr, std::size_t size), (override));
};

} // namespace

class CpuRuntimeManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    manager_ = std::make_unique<cpu_rt::CpuRuntimeManager>();
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
  }

  std::unique_ptr<cpu_rt::CpuRuntimeManager> manager_;
};

TEST_F(CpuRuntimeManagerTest, ConfigureWithDefaultOps) {
  EXPECT_FALSE(manager_->isConfigured());

  manager_->configure();

  EXPECT_TRUE(manager_->isConfigured());
  EXPECT_NE(manager_->slowOps(), nullptr);
  EXPECT_TRUE(manager_->deviceManager().isConfiguredForTest());
}

TEST_F(CpuRuntimeManagerTest, ShutdownClearsState) {
  manager_->configure();
  EXPECT_TRUE(manager_->isConfigured());

  manager_->shutdown();

  EXPECT_FALSE(manager_->isConfigured());
}

TEST_F(CpuRuntimeManagerTest, DeviceManagerReturnsCorrectArch) {
  manager_->configure();

  auto &device_manager = manager_->deviceManager();
  auto lease = device_manager.acquire(base::DeviceHandle{0});
  EXPECT_TRUE(lease);

  // Access arch through lease
  auto arch = lease.payloadPtr()->arch;
  EXPECT_EQ(arch, architecture::detectCpuArchitecture());
}

TEST_F(CpuRuntimeManagerTest, ConfigureWithCustomOps) {
  auto *mock_ops = new NiceMock<CpuSlowOpsMock>();

  ON_CALL(*mock_ops, getDeviceCount()).WillByDefault(Return(1));
  ON_CALL(*mock_ops, detectArchitecture(base::DeviceHandle{0}))
      .WillByDefault(Return(architecture::Architecture::CpuZen4));

  cpu_rt::CpuRuntimeManager::Config config{};
  config.slow_ops = mock_ops;
  manager_->configure(config);

  EXPECT_TRUE(manager_->isConfigured());
  EXPECT_EQ(manager_->slowOps(), mock_ops);
}

TEST_F(CpuRuntimeManagerTest, DoubleConfigureUsesExistingOps) {
  manager_->configure();
  auto *first_ops = manager_->slowOps();

  manager_->configure(); // Should reuse existing ops

  EXPECT_EQ(manager_->slowOps(), first_ops);
}

TEST_F(CpuRuntimeManagerTest, ReconfigureAfterShutdown) {
  manager_->configure();
  manager_->shutdown();

  manager_->configure();

  EXPECT_TRUE(manager_->isConfigured());
  EXPECT_TRUE(manager_->deviceManager().isConfiguredForTest());
}

TEST_F(CpuRuntimeManagerTest, DeviceManagerIsAlive) {
  manager_->configure();

  auto &device_manager = manager_->deviceManager();
  EXPECT_TRUE(device_manager.isAliveForTest(base::DeviceHandle{0}));
}
