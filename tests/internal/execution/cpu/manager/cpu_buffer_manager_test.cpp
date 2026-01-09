#include "orteaf/internal/execution/cpu/manager/cpu_buffer_manager.h"
#include "orteaf/internal/execution/cpu/platform/cpu_slow_ops.h"
#include <gtest/gtest.h>
#include <system_error>

#include <memory>

namespace cpu_rt = orteaf::internal::execution::cpu::manager;
namespace cpu_platform = orteaf::internal::execution::cpu::platform;

class CpuBufferManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    slow_ops_ = std::make_unique<cpu_platform::CpuSlowOpsImpl>();
    manager_ = std::make_unique<cpu_rt::CpuBufferManager>();
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
    slow_ops_.reset();
  }

  void configureManager() {
    cpu_rt::CpuBufferManager::Config config{};
    config.ops = slow_ops_.get();
    manager_->configure(config);
  }

  std::unique_ptr<cpu_platform::CpuSlowOpsImpl> slow_ops_;
  std::unique_ptr<cpu_rt::CpuBufferManager> manager_;
};

TEST_F(CpuBufferManagerTest, ConfigureSucceeds) {
  configureManager();
  EXPECT_TRUE(manager_->isConfiguredForTest());
}

TEST_F(CpuBufferManagerTest, ShutdownClearsState) {
  configureManager();
  manager_->shutdown();
  EXPECT_FALSE(manager_->isConfiguredForTest());
}

TEST_F(CpuBufferManagerTest, AcquireReturnsValidLease) {
  configureManager();

  auto lease = manager_->acquire(1024);
  EXPECT_TRUE(lease);
  EXPECT_NE(lease.payloadPtr(), nullptr);
  EXPECT_NE(lease.payloadPtr()->data, nullptr);
  EXPECT_EQ(lease.payloadPtr()->size, 1024u);
}

TEST_F(CpuBufferManagerTest, AcquireWithAlignmentSucceeds) {
  configureManager();

  constexpr std::size_t kAlignment = 64;
  auto lease = manager_->acquire(512, kAlignment);
  EXPECT_TRUE(lease);
  EXPECT_NE(lease.payloadPtr()->data, nullptr);

  // Verify alignment
  auto ptr = reinterpret_cast<std::uintptr_t>(lease.payloadPtr()->data);
  EXPECT_EQ(ptr % kAlignment, 0u);
}

TEST_F(CpuBufferManagerTest, AcquireZeroSizeThrows) {
  configureManager();
  EXPECT_THROW(manager_->acquire(0), std::system_error);
}

TEST_F(CpuBufferManagerTest, MultipleAcquiresSucceed) {
  configureManager();

  auto lease1 = manager_->acquire(128);
  auto lease2 = manager_->acquire(256);
  auto lease3 = manager_->acquire(512);

  EXPECT_TRUE(lease1);
  EXPECT_TRUE(lease2);
  EXPECT_TRUE(lease3);

  // All should have different data pointers
  EXPECT_NE(lease1.payloadPtr()->data, lease2.payloadPtr()->data);
  EXPECT_NE(lease2.payloadPtr()->data, lease3.payloadPtr()->data);
  EXPECT_NE(lease1.payloadPtr()->data, lease3.payloadPtr()->data);
}

TEST_F(CpuBufferManagerTest, LeaseReleaseDecreasesRefCount) {
  configureManager();

  auto lease = manager_->acquire(1024);
  EXPECT_TRUE(lease);
  EXPECT_EQ(lease.strongCount(), 1u);

  // Copy lease increases ref count
  auto lease_copy = lease;
  EXPECT_EQ(lease.strongCount(), 2u);
  EXPECT_EQ(lease_copy.strongCount(), 2u);

  // Release one decreases ref count
  lease_copy.release();
  EXPECT_EQ(lease.strongCount(), 1u);
}

TEST_F(CpuBufferManagerTest, BufferViewFromLeaseIsValid) {
  configureManager();

  auto lease = manager_->acquire(1024);
  auto view = lease.payloadPtr()->view();

  EXPECT_TRUE(view);
  EXPECT_EQ(view.size(), 1024u);
  EXPECT_EQ(view.offset(), 0u);
  EXPECT_EQ(view.data(), lease.payloadPtr()->data);
}

TEST_F(CpuBufferManagerTest, NotConfiguredThrows) {
  EXPECT_THROW(manager_->acquire(1024), std::system_error);
}
