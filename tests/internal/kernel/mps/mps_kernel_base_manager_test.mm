#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <orteaf/internal/kernel/mps/mps_kernel_base_manager.h>

namespace mps_kernel = orteaf::internal::kernel::mps;

namespace {

class MpsKernelBaseManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    mps_kernel::MpsKernelBaseManager::Config config{};
    config.control_block_capacity = 10;
    config.control_block_block_size = 10;
    config.control_block_growth_chunk_size = 5;
    config.payload_capacity = 10;
    config.payload_block_size = 10;
    config.payload_growth_chunk_size = 5;

    manager_.configure(config);
  }

  void TearDown() override { manager_.shutdown(); }

  mps_kernel::MpsKernelBaseManager manager_;
};

TEST_F(MpsKernelBaseManagerTest, Configure) {
#if ORTEAF_ENABLE_TEST
  EXPECT_TRUE(manager_.isConfiguredForTest());
  EXPECT_EQ(manager_.payloadPoolCapacityForTest(), 10u);
  EXPECT_EQ(manager_.controlBlockPoolCapacityForTest(), 10u);
#endif
}

TEST_F(MpsKernelBaseManagerTest, AcquireCreatesNewKernelBase) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel1");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  auto lease = manager_.acquire(key, keys, 1);
  auto *kernel_base = lease.operator->();
  ASSERT_NE(kernel_base, nullptr);
  EXPECT_EQ(kernel_base->kernelCount(), 1u);
}

TEST_F(MpsKernelBaseManagerTest, AcquireWithMultipleKernels) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_multi");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {
      {"lib1", "func1"}, {"lib2", "func2"}, {"lib3", "func3"}};

  auto lease = manager_.acquire(key, keys, 3);
  auto *kernel_base = lease.operator->();
  ASSERT_NE(kernel_base, nullptr);
  EXPECT_EQ(kernel_base->kernelCount(), 3u);
}

TEST_F(MpsKernelBaseManagerTest, AcquireReturnsCachedLease) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_cached");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  auto lease1 = manager_.acquire(key, keys, 1);
  ASSERT_NE(lease1.operator->(), nullptr);
  auto handle1 = lease1.payloadHandle();

  // Acquire again with same key should return cached lease
  auto lease2 = manager_.acquire(key, keys, 1);
  ASSERT_NE(lease2.operator->(), nullptr);
  auto handle2 = lease2.payloadHandle();

  EXPECT_EQ(handle1.index, handle2.index);
}

TEST_F(MpsKernelBaseManagerTest, AcquireByHandle) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_by_handle");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  auto lease1 = manager_.acquire(key, keys, 1);
  ASSERT_NE(lease1.operator->(), nullptr);
  auto handle = lease1.payloadHandle();

  // Acquire by handle
  auto lease2 = manager_.acquire(handle);
  ASSERT_NE(lease2.operator->(), nullptr);
  EXPECT_EQ(lease2.payloadHandle().index, handle.index);
}

TEST_F(MpsKernelBaseManagerTest, AcquireWithNullKeysReturnInvalid) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_null");
  auto lease = manager_.acquire(key, nullptr, 0);
  EXPECT_EQ(lease.operator->(), nullptr);
}

TEST_F(MpsKernelBaseManagerTest, AcquireWithZeroCountReturnInvalid) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_zero");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};
  auto lease = manager_.acquire(key, keys, 0);
  EXPECT_EQ(lease.operator->(), nullptr);
}

TEST_F(MpsKernelBaseManagerTest, MultipleKernelBases) {
  mps_kernel::MpsKernelBase::KeyLiteral keys1[] = {{"lib1", "func1"}};
  mps_kernel::MpsKernelBase::KeyLiteral keys2[] = {{"lib2", "func2"}};
  mps_kernel::MpsKernelBase::KeyLiteral keys3[] = {{"lib3", "func3"}};

  auto lease1 =
      manager_.acquire(mps_kernel::KernelBaseKey::Named("kernel1"), keys1, 1);
  auto lease2 =
      manager_.acquire(mps_kernel::KernelBaseKey::Named("kernel2"), keys2, 1);
  auto lease3 =
      manager_.acquire(mps_kernel::KernelBaseKey::Named("kernel3"), keys3, 1);

  EXPECT_NE(lease1.operator->(), nullptr);
  EXPECT_NE(lease2.operator->(), nullptr);
  EXPECT_NE(lease3.operator->(), nullptr);

  EXPECT_NE(lease1.payloadHandle().index, lease2.payloadHandle().index);
  EXPECT_NE(lease2.payloadHandle().index, lease3.payloadHandle().index);
  EXPECT_NE(lease1.payloadHandle().index, lease3.payloadHandle().index);
}

TEST_F(MpsKernelBaseManagerTest, Shutdown) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_shutdown");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  {
    auto lease = manager_.acquire(key, keys, 1);
    EXPECT_NE(lease.operator->(), nullptr);
    // Lease goes out of scope here, but lifetime registry keeps it alive
  }

  manager_.shutdown();

#if ORTEAF_ENABLE_TEST
  EXPECT_FALSE(manager_.isConfiguredForTest());
#endif
}

TEST_F(MpsKernelBaseManagerTest, ReconfigureAfterShutdown) {
  manager_.shutdown();

  mps_kernel::MpsKernelBaseManager::Config config{};
  config.control_block_capacity = 5;
  config.control_block_block_size = 5;
  config.control_block_growth_chunk_size = 2;
  config.payload_capacity = 5;
  config.payload_block_size = 5;
  config.payload_growth_chunk_size = 2;

  manager_.configure(config);

#if ORTEAF_ENABLE_TEST
  EXPECT_TRUE(manager_.isConfiguredForTest());
  EXPECT_EQ(manager_.payloadPoolCapacityForTest(), 5u);
  EXPECT_EQ(manager_.controlBlockPoolCapacityForTest(), 5u);
#endif
}

#if ORTEAF_ENABLE_TEST
TEST_F(MpsKernelBaseManagerTest, IsAliveForTestReturnsTrueForValidHandle) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_alive");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  auto lease = manager_.acquire(key, keys, 1);
  EXPECT_NE(lease.operator->(), nullptr);
  auto handle = lease.payloadHandle();

  EXPECT_TRUE(manager_.isAliveForTest(handle));
}

TEST_F(MpsKernelBaseManagerTest, IsAliveForTestReturnsFalseAfterRelease) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_released");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  mps_kernel::MpsKernelBaseHandle handle;
  {
    auto lease = manager_.acquire(key, keys, 1);
    EXPECT_NE(lease.operator->(), nullptr);
    handle = lease.payloadHandle();
    EXPECT_TRUE(manager_.isAliveForTest(handle));
  }

  // After lease is destroyed, handle should still be alive due to lifetime registry
  EXPECT_TRUE(manager_.isAliveForTest(handle));
}
#endif

TEST_F(MpsKernelBaseManagerTest, LeaseLifetimeIsMaintained) {
  auto key = mps_kernel::KernelBaseKey::Named("kernel_lifetime");
  mps_kernel::MpsKernelBase::KeyLiteral keys[] = {{"lib1", "func1"}};

  auto lease1 = manager_.acquire(key, keys, 1);
  // Get kernel base pointer from first lease
  auto *kernel_base1 = lease1.operator->();
  ASSERT_NE(kernel_base1, nullptr);

  // Drop first lease and acquire again
  lease1 = mps_kernel::MpsKernelBaseManager::KernelBaseLease{};

  auto lease2 = manager_.acquire(key, keys, 1);
  auto *kernel_base2 = lease2.operator->();
  ASSERT_NE(kernel_base2, nullptr);

  // Should be the same object (from lifetime registry)
  EXPECT_EQ(kernel_base1, kernel_base2);
}

} // namespace
