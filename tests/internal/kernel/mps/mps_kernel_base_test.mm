#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>

#include <orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution/mps/resource/mps_kernel_base.h>

namespace mps_kernel = ::orteaf::internal::execution::mps::resource;
namespace mps_manager = orteaf::internal::execution::mps::manager;
namespace mps_exec = orteaf::internal::execution::mps;

namespace {

mps_manager::MpsKernelBaseManager::Config makeKernelBaseManagerConfig() {
  mps_manager::MpsKernelBaseManager::Config config{};
  config.control_block_capacity = 4;
  config.control_block_block_size = 4;
  config.payload_capacity = 4;
  config.payload_block_size = 4;
  return config;
}

::orteaf::internal::base::HeapVector<mps_manager::MpsKernelBaseManager::Key>
makeKeys(std::initializer_list<std::pair<const char *, const char *>> names) {
  ::orteaf::internal::base::HeapVector<mps_manager::MpsKernelBaseManager::Key>
      keys;
  keys.reserve(names.size());
  for (const auto &[library, function] : names) {
    keys.pushBack({mps_manager::LibraryKey::Named(std::string(library)),
                   mps_manager::FunctionKey::Named(std::string(function))});
  }
  return keys;
}

TEST(MpsKernelBaseTest, DefaultConstruction) {
  mps_kernel::MpsKernelBase base;
  EXPECT_EQ(base.kernelCount(), 0u);
}

TEST(MpsKernelBaseTest, AddMultipleKeys) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, AddKey) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease =
      manager.acquire(makeKeys({{"library1", "function1"},
                                {"library2", "function2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ReserveKeys) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"},
                                         {"lib", "func"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 10u);
}

TEST(MpsKernelBaseTest, ConfiguredReturnsFalseForUnconfiguredDevice) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  auto device = mps_exec::MpsDeviceHandle{42};
  EXPECT_FALSE(lease->configured(device));
}

TEST(MpsKernelBaseTest, ConfigureWithMultipleKernels) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 2u);
}

TEST(MpsKernelBaseTest, ConfigureMultipleDevices) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->kernelCount(), 1u);
}

TEST(MpsKernelBaseTest, GetPipelineReturnsNullptrForUnconfiguredDevice) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());

  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}}));
  ASSERT_TRUE(lease);
  auto device = mps_exec::MpsDeviceHandle{42};

  auto pipeline = lease->getPipelineLease(device, 0);
  EXPECT_FALSE(pipeline);
}

#if ORTEAF_ENABLE_TESTING
TEST(MpsKernelBaseTest, TestingHelpers) {
  mps_manager::MpsKernelBaseManager manager;
  manager.configureForTest(makeKernelBaseManagerConfig());
  auto lease = manager.acquire(makeKeys({{"lib1", "func1"}, {"lib2", "func2"}}));
  ASSERT_TRUE(lease);

  EXPECT_EQ(lease->deviceCountForTest(), 0u);

  auto &keys = lease->keysForTest();
  EXPECT_EQ(keys.size(), 2u);
}
#endif

} // namespace
