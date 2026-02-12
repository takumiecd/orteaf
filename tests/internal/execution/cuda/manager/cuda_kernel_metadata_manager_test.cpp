#include "orteaf/internal/execution/cuda/manager/cuda_kernel_metadata_manager.h"

#include <gtest/gtest.h>

#include <system_error>

#include "orteaf/internal/base/heap_vector.h"

#if ORTEAF_ENABLE_CUDA

namespace cuda_rt = orteaf::internal::execution::cuda::manager;
namespace base = orteaf::internal::base;

namespace {

cuda_rt::CudaKernelMetadataManager::Config makeConfig() {
  cuda_rt::CudaKernelMetadataManager::Config config{};
  config.control_block_capacity = 2;
  config.control_block_block_size = 2;
  config.payload_capacity = 2;
  config.payload_block_size = 2;
  return config;
}

} // namespace

TEST(CudaKernelMetadataManagerTest, ConfigureSucceeds) {
  cuda_rt::CudaKernelMetadataManager manager;

  manager.configureForTest(makeConfig());
  EXPECT_TRUE(manager.isConfiguredForTest());
}

TEST(CudaKernelMetadataManagerTest, ShutdownClearsState) {
  cuda_rt::CudaKernelMetadataManager manager;
  manager.configureForTest(makeConfig());
  EXPECT_TRUE(manager.isConfiguredForTest());

  manager.shutdown();
  EXPECT_FALSE(manager.isConfiguredForTest());
}

TEST(CudaKernelMetadataManagerTest, AcquireBeforeConfigureThrows) {
  cuda_rt::CudaKernelMetadataManager manager;
  base::HeapVector<cuda_rt::CudaKernelMetadataManager::Key> keys;

  EXPECT_THROW((void)manager.acquire(keys), std::system_error);
}

TEST(CudaKernelMetadataManagerTest, AcquireCopiesKeysIntoPayload) {
  cuda_rt::CudaKernelMetadataManager manager;
  manager.configureForTest(makeConfig());

  base::HeapVector<cuda_rt::CudaKernelMetadataManager::Key> keys;
  keys.pushBack({cuda_rt::ModuleKey::File("module.bin"), "kernel_a"});
  keys.pushBack({cuda_rt::ModuleKey::Embedded("embedded_a"), "kernel_b"});

  auto lease = manager.acquire(keys);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.operator->(), nullptr);
  ASSERT_EQ(lease->keys().size(), 2u);
  EXPECT_EQ(lease->keys()[0].first.identifier, "module.bin");
  EXPECT_EQ(lease->keys()[0].second, "kernel_a");
  EXPECT_EQ(lease->keys()[1].first.identifier, "embedded_a");
  EXPECT_EQ(lease->keys()[1].second, "kernel_b");
}

#endif // ORTEAF_ENABLE_CUDA
