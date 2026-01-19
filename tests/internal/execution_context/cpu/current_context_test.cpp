#include "orteaf/internal/execution_context/cpu/current_context.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include <gtest/gtest.h>

namespace cpu_context = ::orteaf::internal::execution_context::cpu;
namespace cpu_api = ::orteaf::internal::execution::cpu::api;
namespace cpu_exec = ::orteaf::internal::execution::cpu;

class CpuCurrentContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
    cpu_api::CpuExecutionApi::configure(config);
    cpu_context::reset();
  }

  void TearDown() override {
    cpu_context::reset();
    cpu_api::CpuExecutionApi::shutdown();
  }
};

TEST_F(CpuCurrentContextTest, CurrentContextProvidesDefaultDevice) {
  const auto &ctx = cpu_context::currentContext();
  EXPECT_TRUE(ctx.device);

  auto device = cpu_context::currentDevice();
  EXPECT_TRUE(device);
  EXPECT_EQ(ctx.device.payloadHandle(), device.payloadHandle());
  EXPECT_EQ(device.payloadHandle(), cpu_exec::CpuDeviceHandle{0});
}

TEST_F(CpuCurrentContextTest, SetCurrentContextOverridesState) {
  cpu_context::Context ctx{};
  ctx.device = cpu_api::CpuExecutionApi::acquireDevice(cpu_exec::CpuDeviceHandle{0});

  cpu_context::setCurrentContext(std::move(ctx));

  const auto &current_ctx = cpu_context::currentContext();
  EXPECT_TRUE(current_ctx.device);
  EXPECT_EQ(current_ctx.device.payloadHandle(), cpu_exec::CpuDeviceHandle{0});
}

TEST_F(CpuCurrentContextTest, SetCurrentOverridesState) {
  cpu_context::Context ctx{};
  ctx.device = cpu_api::CpuExecutionApi::acquireDevice(cpu_exec::CpuDeviceHandle{0});

  cpu_context::CurrentContext current{};
  current.current = std::move(ctx);
  cpu_context::setCurrent(std::move(current));

  const auto &current_ctx = cpu_context::currentContext();
  EXPECT_TRUE(current_ctx.device);
  EXPECT_EQ(current_ctx.device.payloadHandle(), cpu_exec::CpuDeviceHandle{0});
}

TEST_F(CpuCurrentContextTest, ResetReacquiresDefaultDevice) {
  auto first = cpu_context::currentDevice();
  EXPECT_TRUE(first);

  cpu_context::reset();

  auto second = cpu_context::currentDevice();
  EXPECT_TRUE(second);
  EXPECT_EQ(second.payloadHandle(), cpu_exec::CpuDeviceHandle{0});
}
