#include <gtest/gtest.h>
#include "orteaf/internal/runtime/manager/cpu/device_manager.h"
#include "orteaf/internal/architecture/architecture.h"

using namespace orteaf::internal::runtime;
using namespace orteaf::internal::runtime::cpu;
using namespace orteaf::internal::architecture;

class CpuDeviceManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        CpuDeviceManager.initialize_devices();
    }
    void TearDown() override {
        CpuDeviceManager.shutdown();
    }
};

TEST_F(CpuDeviceManagerTest, GetDeviceCount) {
    EXPECT_EQ(CpuDeviceManager.get_device_count(), 1);
}

TEST_F(CpuDeviceManagerTest, GetArchitecture) {
    DeviceId device_id{0};
    EXPECT_EQ(CpuDeviceManager.get_arch(device_id), Architecture::cpu_generic);
}

TEST_F(CpuDeviceManagerTest, GetIsAlive) {
    DeviceId device_id{0};
    EXPECT_TRUE(CpuDeviceManager.is_alive(device_id));
}