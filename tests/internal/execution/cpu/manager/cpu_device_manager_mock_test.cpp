#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <system_error>

#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"

namespace architecture = orteaf::internal::architecture;
namespace cpu = orteaf::internal::execution::cpu;
namespace cpu_rt = orteaf::internal::execution::cpu::manager;
namespace cpu_platform = orteaf::internal::execution::cpu::platform;
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
              (cpu::CpuDeviceHandle device_id), (override));
  MOCK_METHOD(void *, allocBuffer, (std::size_t size, std::size_t alignment),
              (override));
  MOCK_METHOD(void, deallocBuffer, (void *ptr, std::size_t size), (override));
};

} // namespace

class CpuDeviceManagerMockTest : public ::testing::Test {
protected:
  void SetUp() override {
    mock_ops_ = std::make_unique<NiceMock<CpuSlowOpsMock>>();
    manager_ = std::make_unique<cpu_rt::CpuDeviceManager>();
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
    mock_ops_.reset();
  }

  void configureManager() {
    // Setup default mock behavior
    ON_CALL(*mock_ops_, getDeviceCount()).WillByDefault(Return(1));
    ON_CALL(*mock_ops_, detectArchitecture(cpu::CpuDeviceHandle{0}))
        .WillByDefault(Return(architecture::Architecture::CpuZen4));

    cpu_rt::CpuDeviceManager::Config config{};
    manager_->configureForTest(config, mock_ops_.get());
  }

  std::unique_ptr<NiceMock<CpuSlowOpsMock>> mock_ops_;
  std::unique_ptr<cpu_rt::CpuDeviceManager> manager_;
};

TEST_F(CpuDeviceManagerMockTest, ConfiguresWithMockedArch) {
  EXPECT_FALSE(manager_->isConfiguredForTest());

  EXPECT_CALL(*mock_ops_, detectArchitecture(cpu::CpuDeviceHandle{0}))
      .WillOnce(Return(architecture::Architecture::CpuZen4));

  configureManager();

  EXPECT_TRUE(manager_->isConfiguredForTest());
  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  EXPECT_TRUE(lease);
  // Access arch through lease
  EXPECT_EQ(lease->arch, architecture::Architecture::CpuZen4);
  EXPECT_TRUE(manager_->isAliveForTest(cpu::CpuDeviceHandle{0}));
}

TEST_F(CpuDeviceManagerMockTest, ShutdownClearsState) {
  EXPECT_CALL(*mock_ops_, detectArchitecture(cpu::CpuDeviceHandle{0}))
      .WillOnce(Return(architecture::Architecture::CpuZen4));

  configureManager();
  manager_->shutdown();

  EXPECT_FALSE(manager_->isConfiguredForTest());
  EXPECT_FALSE(manager_->isAliveForTest(cpu::CpuDeviceHandle{0}));
}

TEST_F(CpuDeviceManagerMockTest, InvalidDeviceIdThrows) {
  EXPECT_CALL(*mock_ops_, detectArchitecture(cpu::CpuDeviceHandle{0}))
      .WillOnce(Return(architecture::Architecture::CpuZen4));

  configureManager();

  // Device handle 1 is invalid for CPU (only 0 is valid)
  EXPECT_THROW(manager_->acquire(cpu::CpuDeviceHandle{1}), std::system_error);
}
