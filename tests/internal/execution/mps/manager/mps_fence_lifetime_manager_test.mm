/**
 * @file mps_fence_lifetime_manager_test.mm
 * @brief Tests for MpsFenceLifetimeManager setter behavior.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "orteaf/internal/execution/mps/manager/mps_fence_lifetime_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_fence_manager.h"
#include "tests/internal/execution/mps/manager/testing/execution_mock.h"
#include "tests/internal/testing/error_assert.h"

namespace mps_mgr = orteaf::internal::execution::mps::manager;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace base = orteaf::internal::base;

#if ORTEAF_ENABLE_MPS

class MpsFenceLifetimeManagerTest : public ::testing::Test {
protected:
  using MockOps = ::orteaf::tests::execution::mps::MpsExecutionOpsMock;

  void SetUp() override {
    device_ = reinterpret_cast<mps_wrapper::MpsDevice_t>(0x1);
    fence_handle_ = reinterpret_cast<mps_wrapper::MpsFence_t>(0x2);

    EXPECT_CALL(ops_, createFence(device_))
        .WillOnce(::testing::Return(fence_handle_));

    mps_mgr::MpsFenceManager::Config config{};
    config.device = device_;
    config.ops = &ops_;
    config.pool.payload_capacity = 1;
    config.pool.control_block_capacity = 1;
    config.pool.payload_block_size = 1;
    config.pool.control_block_block_size = 1;
    config.pool.payload_growth_chunk_size = 1;
    config.pool.control_block_growth_chunk_size = 1;
    fence_manager_.configure(config);
  }

  void TearDown() override {
    lifetime_manager_.clear();
    EXPECT_CALL(ops_, destroyFence(fence_handle_)).Times(1);
    fence_manager_.shutdown();
  }

  ::testing::NiceMock<MockOps> ops_{};
  mps_mgr::MpsFenceManager fence_manager_{};
  mps_mgr::MpsFenceLifetimeManager lifetime_manager_{};
  mps_wrapper::MpsDevice_t device_{nullptr};
  mps_wrapper::MpsFence_t fence_handle_{nullptr};
};

TEST_F(MpsFenceLifetimeManagerTest, SettersAllowRebindWhenEmpty) {
  mps_mgr::MpsFenceManager fence_manager_b;
  const base::CommandQueueHandle handle_a{1};
  const base::CommandQueueHandle handle_b{2};

  EXPECT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  EXPECT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_b));
  EXPECT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_a));
  EXPECT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_b));
}

TEST_F(MpsFenceLifetimeManagerTest,
       SettersRejectDifferentValuesWhenNotEmpty) {
  const base::CommandQueueHandle handle_a{3};
  const base::CommandQueueHandle handle_b{4};
  auto *command_buffer =
      reinterpret_cast<mps_wrapper::MpsCommandBuffer_t>(0x9);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_a));

  auto lease = lifetime_manager_.acquire(command_buffer);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.payloadPtr(), nullptr);
  EXPECT_EQ(lease.payloadPtr()->commandBuffer(), command_buffer);

  mps_mgr::MpsFenceManager fence_manager_b;
  EXPECT_FALSE(lifetime_manager_.setFenceManager(&fence_manager_b));
  EXPECT_FALSE(lifetime_manager_.setCommandQueueHandle(handle_b));

  EXPECT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  EXPECT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_a));
}

TEST_F(MpsFenceLifetimeManagerTest, AcquireRequiresCommandBuffer) {
  const base::CommandQueueHandle handle{5};

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      [&] { (void)lifetime_manager_.acquire(nullptr); });
}

TEST_F(MpsFenceLifetimeManagerTest, AcquireBindsCommandBufferAndTracks) {
  const base::CommandQueueHandle handle{6};
  auto *command_buffer =
      reinterpret_cast<mps_wrapper::MpsCommandBuffer_t>(0xA);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  auto lease = lifetime_manager_.acquire(command_buffer);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.payloadPtr(), nullptr);
  EXPECT_EQ(lease.payloadPtr()->commandBuffer(), command_buffer);
  EXPECT_EQ(lifetime_manager_.size(), 1u);
}

#endif // ORTEAF_ENABLE_MPS
