#include "orteaf/user/execution_context/cpu_context_guard.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"
#include <gtest/gtest.h>

namespace user_ctx = ::orteaf::user::execution_context;
namespace cpu_api = ::orteaf::internal::execution::cpu::api;
namespace cpu_exec = ::orteaf::internal::execution::cpu;
namespace cpu_context = ::orteaf::internal::execution_context::cpu;

class CpuExecutionContextGuardTest : public ::testing::Test {
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

TEST_F(CpuExecutionContextGuardTest, GuardRestoresPreviousContext) {
  const auto baseline = cpu_context::currentDevice().payloadHandle();

  {
    user_ctx::CpuExecutionContextGuard guard;
    auto active = cpu_context::currentDevice().payloadHandle();
    EXPECT_EQ(active, cpu_exec::CpuDeviceHandle{0});
  }

  const auto restored = cpu_context::currentDevice().payloadHandle();
  EXPECT_EQ(restored, baseline);
}

TEST_F(CpuExecutionContextGuardTest, GuardWithExplicitDevice) {
  user_ctx::CpuExecutionContextGuard guard(cpu_exec::CpuDeviceHandle{0});
  auto active = cpu_context::currentDevice().payloadHandle();
  EXPECT_EQ(active, cpu_exec::CpuDeviceHandle{0});
}

TEST_F(CpuExecutionContextGuardTest, GuardMoveTransfersOwnership) {
  const auto baseline = cpu_context::currentDevice().payloadHandle();

  {
    user_ctx::CpuExecutionContextGuard guard;
    user_ctx::CpuExecutionContextGuard moved(std::move(guard));
    (void)moved;
  }

  const auto restored = cpu_context::currentDevice().payloadHandle();
  EXPECT_EQ(restored, baseline);
}
