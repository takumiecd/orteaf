#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cpu_detect.h"
#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"
#include <gtest/gtest.h>
#include <system_error>

#include <cstdlib>
#include <memory>

namespace cpu = orteaf::internal::execution::cpu;
namespace cpu_rt = orteaf::internal::execution::cpu::manager;
namespace cpu_platform = orteaf::internal::execution::cpu::platform;
namespace architecture = orteaf::internal::architecture;

class CpuDeviceManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    slow_ops_ = std::make_unique<cpu_platform::CpuSlowOpsImpl>();
    manager_ = std::make_unique<cpu_rt::CpuDeviceManager>();
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
    slow_ops_.reset();
  }

  void configureManager() {
    cpu_rt::CpuDeviceManager::Config config{};
    manager_->configureForTest(config, slow_ops_.get());
  }

  std::unique_ptr<cpu_platform::CpuSlowOpsImpl> slow_ops_;
  std::unique_ptr<cpu_rt::CpuDeviceManager> manager_;
};

TEST_F(CpuDeviceManagerTest, ConfigurePopulatesState) {
  EXPECT_FALSE(manager_->isConfiguredForTest());

  configureManager();
  EXPECT_TRUE(manager_->isConfiguredForTest());
  EXPECT_TRUE(manager_->isAliveForTest(cpu::CpuDeviceHandle{0}));
}

TEST_F(CpuDeviceManagerTest, AcquireReturnsValidLease) {
  configureManager();

  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  EXPECT_TRUE(lease);
  EXPECT_NE(lease.payloadPtr(), nullptr);
}

TEST_F(CpuDeviceManagerTest, LeaseContainsCorrectArch) {
  configureManager();

  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  EXPECT_TRUE(lease);

  // Access arch through lease
  auto arch = lease.payloadPtr()->arch;
  EXPECT_EQ(arch, architecture::detectCpuArchitecture());
}

TEST_F(CpuDeviceManagerTest, ShutdownClearsState) {
  configureManager();
  manager_->shutdown();

  EXPECT_FALSE(manager_->isConfiguredForTest());
}

#define ORTEAF_CPU_ENV_VAR "ORTEAF_EXPECT_CPU_MANAGER_ARCH"

/// Manual test hook: set ORTEAF_EXPECT_CPU_MANAGER_ARCH=AppleM4Pro (など)
/// で検証。
TEST_F(CpuDeviceManagerTest, ManualEnvironmentCheck) {
  const char *expected_env = std::getenv(ORTEAF_CPU_ENV_VAR);
  if (!expected_env) {
    GTEST_SKIP() << "Set " ORTEAF_CPU_ENV_VAR
                    " to run this test on your environment.";
  }
  configureManager();
  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  const auto arch = lease.payloadPtr()->arch;
  EXPECT_STREQ(expected_env, architecture::idOf(arch).data());
}

TEST_F(CpuDeviceManagerTest, GetArchitectureMatchesDetector) {
  configureManager();
  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  EXPECT_EQ(lease.payloadPtr()->arch, architecture::detectCpuArchitecture());
}

TEST_F(CpuDeviceManagerTest, IsAliveReflectsInitialization) {
  configureManager();
  EXPECT_TRUE(manager_->isAliveForTest(cpu::CpuDeviceHandle{0}));
  manager_->shutdown();
  EXPECT_FALSE(manager_->isAliveForTest(cpu::CpuDeviceHandle{0}));
}

TEST_F(CpuDeviceManagerTest, InvalidDeviceHandleThrows) {
  configureManager();
  EXPECT_THROW(manager_->acquire(cpu::CpuDeviceHandle{1}), std::system_error);
}

TEST_F(CpuDeviceManagerTest, LeaseReleaseWorks) {
  configureManager();

  auto lease = manager_->acquire(cpu::CpuDeviceHandle{0});
  EXPECT_TRUE(lease);
  // Device manager keeps a copy in lifetime registry, so count is 2
  EXPECT_EQ(lease.strongCount(), 2u);

  // Copy increases ref count
  auto lease_copy = lease;
  EXPECT_EQ(lease.strongCount(), 3u);

  // Release through lease method
  lease_copy.release();
  EXPECT_EQ(lease.strongCount(), 2u);
}
