#if ORTEAF_ENABLE_MPS

#include "orteaf/user/execution_context/mps_context_guard.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution_context/mps/current_context.h"
#include <gtest/gtest.h>

#include <string>

namespace user_ctx = ::orteaf::user::execution_context;
namespace mps_api = ::orteaf::internal::execution::mps::api;
namespace mps_platform = ::orteaf::internal::execution::mps::platform;
namespace mps_exec = ::orteaf::internal::execution::mps;
namespace mps_context = ::orteaf::internal::execution_context::mps;

namespace {

bool configureMps(std::string *error) {
  auto *ops = new mps_platform::MpsSlowOpsImpl();
  const int device_count = ops->getDeviceCount();
  if (device_count <= 0) {
    delete ops;
    if (error) {
      *error = "No MPS devices available";
    }
    return false;
  }

  mps_api::MpsExecutionApi::ExecutionManager::Config config{};
  config.slow_ops = ops;

  const auto capacity = static_cast<std::size_t>(device_count);
  auto &device_cfg = config.device_config;
  device_cfg.control_block_capacity = capacity;
  device_cfg.control_block_block_size = capacity;
  device_cfg.control_block_growth_chunk_size = 1;
  device_cfg.payload_capacity = capacity;
  device_cfg.payload_block_size = capacity;
  device_cfg.payload_growth_chunk_size = 1;

  auto configure_pool = [](auto &cfg, std::size_t pool_capacity) {
    cfg.control_block_capacity = pool_capacity;
    cfg.control_block_block_size = pool_capacity;
    cfg.control_block_growth_chunk_size = 1;
    cfg.payload_capacity = pool_capacity;
    cfg.payload_block_size = pool_capacity;
    cfg.payload_growth_chunk_size = 1;
  };

  configure_pool(device_cfg.command_queue_config, 2);
  configure_pool(device_cfg.event_config, 1);
  configure_pool(device_cfg.fence_config, 1);
  configure_pool(device_cfg.heap_config, 1);
  configure_pool(device_cfg.library_config, 1);
  configure_pool(device_cfg.graph_config, 1);

  try {
    mps_api::MpsExecutionApi::configure(config);
  } catch (const std::exception &ex) {
    delete ops;
    if (error) {
      *error = ex.what();
    }
    return false;
  }

  return true;
}

} // namespace

class MpsExecutionContextGuardTest : public ::testing::Test {
protected:
  void TearDown() override {
    mps_context::reset();
    mps_api::MpsExecutionApi::shutdown();
  }
};

TEST_F(MpsExecutionContextGuardTest, GuardRestoresPreviousContext) {
  std::string error;
  if (!configureMps(&error)) {
    GTEST_SKIP() << error;
  }
  mps_context::reset();

  const auto baseline_queue = mps_context::currentCommandQueue().payloadHandle();
  auto device_lease = mps_context::currentDevice();
  if (!device_lease) {
    GTEST_SKIP() << "Failed to acquire MPS device";
  }
  auto *resource = device_lease.operator->();
  if (resource == nullptr) {
    GTEST_SKIP() << "Failed to access MPS device resource";
  }
  auto alt_queue = resource->command_queue_manager.acquire();
  if (!alt_queue) {
    GTEST_SKIP() << "Failed to acquire alternate MPS command queue";
  }
  const auto alt_handle = alt_queue.payloadHandle();
  if (alt_handle == baseline_queue) {
    GTEST_SKIP() << "Only one command queue handle available";
  }

  {
    user_ctx::MpsExecutionContextGuard guard(mps_exec::MpsDeviceHandle{0},
                                             alt_handle);
    auto active = mps_context::currentCommandQueue().payloadHandle();
    EXPECT_EQ(active, alt_handle);
  }

  const auto restored = mps_context::currentCommandQueue().payloadHandle();
  EXPECT_EQ(restored, baseline_queue);
}

TEST_F(MpsExecutionContextGuardTest, GuardWithDefaultContext) {
  std::string error;
  if (!configureMps(&error)) {
    GTEST_SKIP() << error;
  }
  mps_context::reset();

  user_ctx::MpsExecutionContextGuard guard;
  auto active_device = mps_context::currentDevice().payloadHandle();
  EXPECT_EQ(active_device, mps_exec::MpsDeviceHandle{0});
}

#endif // ORTEAF_ENABLE_MPS
