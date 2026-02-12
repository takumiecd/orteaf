#include "orteaf/internal/execution/cuda/manager/cuda_kernel_base_manager.h"

#include <gtest/gtest.h>

#include <system_error>

#include "orteaf/internal/base/heap_vector.h"

#if ORTEAF_ENABLE_CUDA

namespace cuda_rt = orteaf::internal::execution::cuda::manager;
namespace base = orteaf::internal::base;
namespace kernel = orteaf::internal::kernel;

namespace {

cuda_rt::CudaKernelBaseManager::Config makeConfig() {
  cuda_rt::CudaKernelBaseManager::Config config{};
  config.control_block_capacity = 2;
  config.control_block_block_size = 2;
  config.payload_capacity = 2;
  config.payload_block_size = 2;
  return config;
}

void dummyExecute(
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase &,
    kernel::KernelArgs &) {}

} // namespace

TEST(CudaKernelBaseManagerTest, ConfigureSucceeds) {
  cuda_rt::CudaKernelBaseManager manager;

  manager.configureForTest(makeConfig());
  EXPECT_TRUE(manager.isConfiguredForTest());
}

TEST(CudaKernelBaseManagerTest, ShutdownClearsState) {
  cuda_rt::CudaKernelBaseManager manager;
  manager.configureForTest(makeConfig());
  EXPECT_TRUE(manager.isConfiguredForTest());

  manager.shutdown();
  EXPECT_FALSE(manager.isConfiguredForTest());
}

TEST(CudaKernelBaseManagerTest, AcquireBeforeConfigureThrows) {
  cuda_rt::CudaKernelBaseManager manager;
  base::HeapVector<cuda_rt::CudaKernelBaseManager::Key> keys;

  EXPECT_THROW((void)manager.acquire(keys), std::system_error);
}

TEST(CudaKernelBaseManagerTest, AcquireCopiesKeysIntoPayload) {
  cuda_rt::CudaKernelBaseManager manager;
  manager.configureForTest(makeConfig());

  base::HeapVector<cuda_rt::CudaKernelBaseManager::Key> keys;
  keys.pushBack({cuda_rt::ModuleKey::File("module.bin"), "kernel_a"});
  keys.pushBack({cuda_rt::ModuleKey::Embedded("embedded_a"), "kernel_b"});

  auto lease = manager.acquire(keys);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.operator->(), nullptr);
  EXPECT_EQ(lease->kernelCount(), 2u);
  ASSERT_EQ(lease->keys().size(), 2u);
  EXPECT_EQ(lease->keys()[0].first.identifier, "module.bin");
  EXPECT_EQ(lease->keys()[0].second, "kernel_a");
  EXPECT_EQ(lease->keys()[1].first.identifier, "embedded_a");
  EXPECT_EQ(lease->keys()[1].second, "kernel_b");
}

TEST(CudaKernelBaseManagerTest, AcquireFromMetadataCopiesKeysAndExecute) {
  cuda_rt::CudaKernelBaseManager manager;
  manager.configureForTest(makeConfig());

  base::HeapVector<cuda_rt::CudaKernelBaseManager::Key> keys;
  keys.pushBack({cuda_rt::ModuleKey::File("module.bin"), "kernel_a"});

  ::orteaf::internal::execution::cuda::resource::CudaKernelMetadata metadata{};
  ASSERT_TRUE(metadata.initialize(keys));
  metadata.setExecute(dummyExecute);

  auto lease = manager.acquire(metadata);
  ASSERT_TRUE(lease);
  ASSERT_NE(lease.operator->(), nullptr);
  ASSERT_EQ(lease->keys().size(), 1u);
  EXPECT_EQ(lease->keys()[0].first.identifier, "module.bin");
  EXPECT_EQ(lease->keys()[0].second, "kernel_a");
  EXPECT_EQ(lease->execute(), dummyExecute);
}

TEST(CudaKernelBaseManagerTest, GetFunctionLeaseReturnsInvalidWhenUnconfigured) {
  cuda_rt::CudaKernelBaseManager manager;
  manager.configureForTest(makeConfig());

  base::HeapVector<cuda_rt::CudaKernelBaseManager::Key> keys;
  keys.pushBack({cuda_rt::ModuleKey::File("module.bin"), "kernel_a"});

  auto base_lease = manager.acquire(keys);
  ASSERT_TRUE(base_lease);

  auto function_lease = base_lease->getFunctionLease(
      ::orteaf::internal::execution::cuda::CudaContextHandle{}, 0);
  EXPECT_FALSE(function_lease);
}

#endif // ORTEAF_ENABLE_CUDA
