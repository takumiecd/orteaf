#include "orteaf/internal/architecture/architecture.h"
#include "orteaf/internal/architecture/cpu_detect.h"
#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"
#include <gtest/gtest.h>
#include <system_error>

#include <cstdlib>
#include <memory>

namespace base = orteaf::internal::base;
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
    config.ops = slow_ops_.get();
    manager_->configure(config);
  }

  std::unique_ptr<cpu_platform::CpuSlowOpsImpl> slow_ops_;
  std::unique_ptr<cpu_rt::CpuDeviceManager> manager_;
};

TEST_F(CpuDeviceManagerTest, ConfigurePopulatesState) {
  EXPECT_EQ(manager_->getDeviceCount(), 0u);

  configureManager();
  EXPECT_EQ(manager_->getDeviceCount(), 1u);
  EXPECT_TRUE(manager_->isAlive(base::DeviceHandle{0}));
}

TEST_F(CpuDeviceManagerTest, AcquireAndGetArch) {
  configureManager();

  auto lease = manager_->acquire(base::DeviceHandle{0});
  EXPECT_TRUE(lease);

  auto arch = manager_->getArch(base::DeviceHandle{0});
  EXPECT_EQ(arch, architecture::detectCpuArchitecture());
}

TEST_F(CpuDeviceManagerTest, ShutdownClearsState) {
  configureManager();
  manager_->shutdown();

  EXPECT_EQ(manager_->getDeviceCount(), 0u);
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
  auto lease = manager_->acquire(base::DeviceHandle{0});
  const auto arch = manager_->getArch(base::DeviceHandle{0});
  EXPECT_STREQ(expected_env, architecture::idOf(arch).data());
}

TEST_F(CpuDeviceManagerTest, GetArchitectureMatchesDetector) {
  configureManager();
  auto lease = manager_->acquire(base::DeviceHandle{0});
  EXPECT_EQ(manager_->getArch(base::DeviceHandle{0}),
            architecture::detectCpuArchitecture());
}

TEST_F(CpuDeviceManagerTest, IsAliveReflectsInitialization) {
  configureManager();
  EXPECT_TRUE(manager_->isAlive(base::DeviceHandle{0}));
  manager_->shutdown();
  EXPECT_FALSE(manager_->isAlive(base::DeviceHandle{0}));
}

TEST_F(CpuDeviceManagerTest, InvalidDeviceHandleThrows) {
  configureManager();
  EXPECT_THROW(manager_->acquire(base::DeviceHandle{1}), std::system_error);
}
