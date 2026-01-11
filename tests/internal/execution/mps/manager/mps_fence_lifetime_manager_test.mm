/**
 * @file mps_fence_lifetime_manager_test.mm
 * @brief Tests for MpsFenceLifetimeManager setter behavior.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>

#include "orteaf/internal/execution/mps/manager/mps_fence_lifetime_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_fence_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "tests/internal/execution/mps/manager/testing/execution_mock.h"
#include "tests/internal/testing/error_assert.h"

namespace mps_mgr = orteaf::internal::execution::mps::manager;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace mps = orteaf::internal::execution::mps;

#if ORTEAF_ENABLE_MPS

namespace {
using CommandBufferHandle =
    orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t;

struct CompletionTracker {
  static void clear() { completed().clear(); }

  static void markCompleted(CommandBufferHandle buffer) {
    completed().insert(buffer);
  }

  static bool isCompleted(CommandBufferHandle buffer) {
    return completed().find(buffer) != completed().end();
  }

private:
  static std::unordered_set<CommandBufferHandle> &completed() {
    static std::unordered_set<CommandBufferHandle> set;
    return set;
  }
};

struct FakeFastOpsControlled {
  static bool isCompleted(CommandBufferHandle buffer) {
    return CompletionTracker::isCompleted(buffer);
  }

  static void waitUntilCompleted(CommandBufferHandle buffer) {
    CompletionTracker::markCompleted(buffer);
  }
};

CommandBufferHandle fakeCommandBuffer(std::uintptr_t value) {
  return reinterpret_cast<CommandBufferHandle>(value);
}
} // namespace

class MpsFenceLifetimeManagerTest : public ::testing::Test {
protected:
  using MockOps = ::orteaf::tests::execution::mps::MpsExecutionOpsMock;

  void SetUp() override {
    CompletionTracker::clear();
    device_ = reinterpret_cast<mps_wrapper::MpsDevice_t>(0x1);
    fence_handle_ = reinterpret_cast<mps_wrapper::MpsFence_t>(0x2);

    EXPECT_CALL(ops_, createFence(device_))
        .WillOnce(::testing::Return(fence_handle_));

    mps_mgr::MpsFenceManager::Config config{};
    config.payload_capacity = 1;
    config.control_block_capacity = 1;
    config.payload_block_size = 1;
    config.control_block_block_size = 1;
    config.payload_growth_chunk_size = 1;
    config.control_block_growth_chunk_size = 1;
    fence_manager_.configureForTest(config, device_, &ops_);
  }

  void TearDown() override {
    if (!lifetime_manager_.empty()) {
      (void)lifetime_manager_.waitUntilReady<FakeFastOpsControlled>();
    }
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
  const mps::MpsCommandQueueHandle handle_a{1};
  const mps::MpsCommandQueueHandle handle_b{2};

  EXPECT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  EXPECT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_b));
  EXPECT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_a));
  EXPECT_TRUE(lifetime_manager_.setCommandQueueHandle(handle_b));
}

TEST_F(MpsFenceLifetimeManagerTest,
       SettersRejectDifferentValuesWhenNotEmpty) {
  const mps::MpsCommandQueueHandle handle_a{3};
  const mps::MpsCommandQueueHandle handle_b{4};
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
  const mps::MpsCommandQueueHandle handle{5};

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      [&] { (void)lifetime_manager_.acquire(nullptr); });
}

TEST_F(MpsFenceLifetimeManagerTest, AcquireBindsCommandBufferAndTracks) {
  const mps::MpsCommandQueueHandle handle{6};
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

TEST_F(MpsFenceLifetimeManagerTest, ReleaseReadyRequiresFenceManager) {
  mps_mgr::MpsFenceLifetimeManager manager;

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      [&] { (void)manager.releaseReady(); });
}

TEST_F(MpsFenceLifetimeManagerTest, ReleaseReadyRequiresQueueHandle) {
  mps_mgr::MpsFenceLifetimeManager manager;
  ASSERT_TRUE(manager.setFenceManager(&fence_manager_));

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      [&] { (void)manager.releaseReady(); });
}

TEST_F(MpsFenceLifetimeManagerTest, ReleaseReadyReturnsZeroWhenNoneCompleted) {
  const mps::MpsCommandQueueHandle handle{7};
  auto command_buffer = fakeCommandBuffer(0x20);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  (void)lifetime_manager_.acquire(command_buffer);

  EXPECT_EQ(lifetime_manager_.releaseReady<FakeFastOpsControlled>(), 0u);
  EXPECT_EQ(lifetime_manager_.size(), 1u);
}

TEST_F(MpsFenceLifetimeManagerTest,
       ReleaseReadyReleasesPrefixAndAdvancesOnCompletion) {
  const mps::MpsCommandQueueHandle handle{8};
  auto command_buffer_a = fakeCommandBuffer(0x30);
  auto command_buffer_b = fakeCommandBuffer(0x31);
  auto command_buffer_c = fakeCommandBuffer(0x32);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0x3);
  auto fence_c = reinterpret_cast<mps_wrapper::MpsFence_t>(0x4);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b))
      .WillOnce(::testing::Return(fence_c));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);
  EXPECT_CALL(ops_, destroyFence(fence_c)).Times(1);

  auto lease_a = lifetime_manager_.acquire(command_buffer_a);
  lease_a.release();
  auto lease_b = lifetime_manager_.acquire(command_buffer_b);
  lease_b.release();
  auto lease_c = lifetime_manager_.acquire(command_buffer_c);

  CompletionTracker::markCompleted(command_buffer_a);
  CompletionTracker::markCompleted(command_buffer_b);

  EXPECT_EQ(lifetime_manager_.releaseReady<FakeFastOpsControlled>(), 2u);
  EXPECT_EQ(lifetime_manager_.size(), 1u);
  ASSERT_NE(lease_c.payloadPtr(), nullptr);
  EXPECT_EQ(lease_c.payloadPtr()->commandBuffer(), command_buffer_c);
  EXPECT_FALSE(lease_c.payloadPtr()->isCompleted());

  CompletionTracker::markCompleted(command_buffer_c);
  EXPECT_EQ(lifetime_manager_.releaseReady<FakeFastOpsControlled>(), 1u);
  EXPECT_EQ(lifetime_manager_.size(), 0u);
  EXPECT_TRUE(lease_c.payloadPtr()->isCompleted());
}

TEST_F(MpsFenceLifetimeManagerTest, ClearThrowsWhenAnyIncomplete) {
  const mps::MpsCommandQueueHandle handle{10};
  auto command_buffer_a = fakeCommandBuffer(0x50);
  auto command_buffer_b = fakeCommandBuffer(0x51);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0x9);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);

  (void)lifetime_manager_.acquire(command_buffer_a);
  (void)lifetime_manager_.acquire(command_buffer_b);

  CompletionTracker::markCompleted(command_buffer_a);

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      [&] { lifetime_manager_.clear<FakeFastOpsControlled>(); });

  EXPECT_EQ(lifetime_manager_.size(), 2u);
  EXPECT_EQ(lifetime_manager_.storageSizeForTest(), 2u);
  EXPECT_EQ(lifetime_manager_.headIndexForTest(), 0u);
}

TEST_F(MpsFenceLifetimeManagerTest, ClearReleasesAllWhenCompleted) {
  const mps::MpsCommandQueueHandle handle{11};
  auto command_buffer_a = fakeCommandBuffer(0x60);
  auto command_buffer_b = fakeCommandBuffer(0x61);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0xA);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);

  auto lease_a = lifetime_manager_.acquire(command_buffer_a);
  auto lease_b = lifetime_manager_.acquire(command_buffer_b);

  CompletionTracker::markCompleted(command_buffer_a);
  CompletionTracker::markCompleted(command_buffer_b);

  lifetime_manager_.clear<FakeFastOpsControlled>();

  EXPECT_EQ(lifetime_manager_.size(), 0u);
  EXPECT_EQ(lifetime_manager_.storageSizeForTest(), 0u);
  ASSERT_NE(lease_a.payloadPtr(), nullptr);
  ASSERT_NE(lease_b.payloadPtr(), nullptr);
  EXPECT_TRUE(lease_a.payloadPtr()->isCompleted());
  EXPECT_TRUE(lease_b.payloadPtr()->isCompleted());
}

TEST_F(MpsFenceLifetimeManagerTest, WaitUntilReadyReleasesAll) {
  const mps::MpsCommandQueueHandle handle{12};
  auto command_buffer_a = fakeCommandBuffer(0x70);
  auto command_buffer_b = fakeCommandBuffer(0x71);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0xB);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);

  auto lease_a = lifetime_manager_.acquire(command_buffer_a);
  auto lease_b = lifetime_manager_.acquire(command_buffer_b);

  EXPECT_EQ(lifetime_manager_.waitUntilReady<FakeFastOpsControlled>(), 2u);
  EXPECT_EQ(lifetime_manager_.size(), 0u);
  EXPECT_EQ(lifetime_manager_.storageSizeForTest(), 0u);
  ASSERT_NE(lease_a.payloadPtr(), nullptr);
  ASSERT_NE(lease_b.payloadPtr(), nullptr);
  EXPECT_TRUE(lease_a.payloadPtr()->isCompleted());
  EXPECT_TRUE(lease_b.payloadPtr()->isCompleted());
}

TEST_F(MpsFenceLifetimeManagerTest,
       ReleaseReadySkipsCompactionWhenHeadBelowHalf) {
  const mps::MpsCommandQueueHandle handle{9};
  auto command_buffer_a = fakeCommandBuffer(0x40);
  auto command_buffer_b = fakeCommandBuffer(0x41);
  auto command_buffer_c = fakeCommandBuffer(0x42);
  auto command_buffer_d = fakeCommandBuffer(0x43);
  auto command_buffer_e = fakeCommandBuffer(0x44);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0x5);
  auto fence_c = reinterpret_cast<mps_wrapper::MpsFence_t>(0x6);
  auto fence_d = reinterpret_cast<mps_wrapper::MpsFence_t>(0x7);
  auto fence_e = reinterpret_cast<mps_wrapper::MpsFence_t>(0x8);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b))
      .WillOnce(::testing::Return(fence_c))
      .WillOnce(::testing::Return(fence_d))
      .WillOnce(::testing::Return(fence_e));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);
  EXPECT_CALL(ops_, destroyFence(fence_c)).Times(1);
  EXPECT_CALL(ops_, destroyFence(fence_d)).Times(1);
  EXPECT_CALL(ops_, destroyFence(fence_e)).Times(1);

  auto lease_a = lifetime_manager_.acquire(command_buffer_a);
  lease_a.release();
  auto lease_b = lifetime_manager_.acquire(command_buffer_b);
  lease_b.release();
  auto lease_c = lifetime_manager_.acquire(command_buffer_c);
  lease_c.release();
  auto lease_d = lifetime_manager_.acquire(command_buffer_d);
  lease_d.release();
  auto lease_e = lifetime_manager_.acquire(command_buffer_e);
  lease_e.release();

  CompletionTracker::markCompleted(command_buffer_a);

  EXPECT_EQ(lifetime_manager_.releaseReady<FakeFastOpsControlled>(), 1u);
  EXPECT_EQ(lifetime_manager_.size(), 4u);
  EXPECT_EQ(lifetime_manager_.storageSizeForTest(), 5u);
  EXPECT_EQ(lifetime_manager_.headIndexForTest(), 1u);
}

#if ORTEAF_MPS_DEBUG_ENABLED
TEST_F(MpsFenceLifetimeManagerTest,
       ReleaseReadyThrowsWhenPrefixNotReadyInDebug) {
  const mps::MpsCommandQueueHandle handle{13};
  auto command_buffer_a = fakeCommandBuffer(0x80);
  auto command_buffer_b = fakeCommandBuffer(0x81);
  auto fence_b = reinterpret_cast<mps_wrapper::MpsFence_t>(0xC);

  ASSERT_TRUE(lifetime_manager_.setFenceManager(&fence_manager_));
  ASSERT_TRUE(lifetime_manager_.setCommandQueueHandle(handle));

  EXPECT_CALL(ops_, createFence(device_))
      .WillOnce(::testing::Return(fence_b));
  EXPECT_CALL(ops_, destroyFence(fence_b)).Times(1);

  (void)lifetime_manager_.acquire(command_buffer_a);
  (void)lifetime_manager_.acquire(command_buffer_b);

  CompletionTracker::markCompleted(command_buffer_b);

  ::orteaf::tests::ExpectError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
      [&] { (void)lifetime_manager_.releaseReady<FakeFastOpsControlled>(); });

  EXPECT_EQ(lifetime_manager_.size(), 2u);
  EXPECT_EQ(lifetime_manager_.storageSizeForTest(), 2u);
  EXPECT_EQ(lifetime_manager_.headIndexForTest(), 0u);
}
#endif

#endif // ORTEAF_ENABLE_MPS
