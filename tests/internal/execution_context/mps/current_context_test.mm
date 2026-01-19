#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution_context/mps/current_context.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include <gtest/gtest.h>

#include <string>

namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_api = ::orteaf::internal::execution::mps::api;
namespace mps_platform = ::orteaf::internal::execution::mps::platform;
namespace mps_exec = ::orteaf::internal::execution::mps;

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

  auto configure_pool = [](auto &cfg) {
    cfg.control_block_capacity = 1;
    cfg.control_block_block_size = 1;
    cfg.control_block_growth_chunk_size = 1;
    cfg.payload_capacity = 1;
    cfg.payload_block_size = 1;
    cfg.payload_growth_chunk_size = 1;
  };

  configure_pool(device_cfg.command_queue_config);
  configure_pool(device_cfg.event_config);
  configure_pool(device_cfg.fence_config);
  configure_pool(device_cfg.heap_config);
  configure_pool(device_cfg.library_config);
  configure_pool(device_cfg.graph_config);

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

class MpsCurrentContextTest : public ::testing::Test {
protected:
  void TearDown() override {
    mps_context::reset();
    mps_api::MpsExecutionApi::shutdown();
  }
};

TEST_F(MpsCurrentContextTest, CurrentContextProvidesDefaultResources) {
  std::string error;
  if (!configureMps(&error)) {
    GTEST_SKIP() << error;
  }
  mps_context::reset();

  const auto &ctx = mps_context::currentContext();
  EXPECT_TRUE(ctx.device);

  auto device = mps_context::currentDevice();
  EXPECT_TRUE(device);
  EXPECT_EQ(ctx.device.payloadHandle(), device.payloadHandle());

  auto queue = mps_context::currentCommandQueue();
  EXPECT_TRUE(queue);
  EXPECT_EQ(ctx.command_queue.payloadHandle(), queue.payloadHandle());
  EXPECT_EQ(device.payloadHandle(), mps_exec::MpsDeviceHandle{0});
}

TEST_F(MpsCurrentContextTest, SetCurrentContextOverridesState) {
  std::string error;
  if (!configureMps(&error)) {
    GTEST_SKIP() << error;
  }
  mps_context::reset();

  mps_context::Context ctx{};
  ctx.device = mps_api::MpsExecutionApi::acquireDevice(mps_exec::MpsDeviceHandle{0});
  auto *resource = ctx.device.operator->();
  if (resource == nullptr) {
    GTEST_SKIP() << "Failed to acquire MPS device";
  }
  ctx.command_queue = resource->command_queue_manager.acquire();
  if (!ctx.command_queue) {
    GTEST_SKIP() << "Failed to acquire MPS command queue";
  }

  mps_context::setCurrentContext(std::move(ctx));

  const auto &current_ctx = mps_context::currentContext();
  EXPECT_TRUE(current_ctx.device);
  EXPECT_TRUE(current_ctx.command_queue);
  EXPECT_EQ(current_ctx.device.payloadHandle(), mps_exec::MpsDeviceHandle{0});
}

#endif // ORTEAF_ENABLE_MPS
