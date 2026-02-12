#include <gtest/gtest.h>

#include <string>
#include <variant>

#if ORTEAF_ENABLE_CUDA
#include "orteaf/internal/execution/cuda/api/cuda_execution_api.h"
#include "orteaf/internal/execution/cuda/platform/cuda_slow_ops.h"
#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"
#endif
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace kernel = orteaf::internal::kernel;

namespace {

TEST(KernelMetadataTest, DefaultConstructionIsEmpty) {
  kernel::core::KernelMetadataLease metadata;
  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(metadata.lease()));

  auto entry = metadata.rebuild();
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entry.base()));
}

TEST(KernelMetadataTest, SetLeaseMonostateKeepsEmpty) {
  kernel::core::KernelMetadataLease metadata;
  metadata.setLease(kernel::core::KernelMetadataLease::Variant{});

  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(metadata.lease()));
  auto entry = metadata.rebuild();
  EXPECT_TRUE(std::holds_alternative<std::monostate>(entry.base()));
}

#if ORTEAF_ENABLE_CUDA

namespace cuda_api = orteaf::internal::execution::cuda::api;
namespace cuda_platform = orteaf::internal::execution::cuda::platform;
namespace cuda_wrapper = orteaf::internal::execution::cuda::platform::wrapper;

class TestCudaSlowOps final : public cuda_platform::CudaSlowOps {
public:
  int getDeviceCount() override { return 1; }

  cuda_wrapper::CudaDevice_t getDevice(std::uint32_t) override {
    return cuda_wrapper::CudaDevice_t{0};
  }

  cuda_wrapper::ComputeCapability getComputeCapability(
      cuda_wrapper::CudaDevice_t) override {
    return cuda_wrapper::ComputeCapability{8, 0};
  }

  std::string getDeviceName(cuda_wrapper::CudaDevice_t) override {
    return "mock-cuda";
  }

  std::string getDeviceVendor(cuda_wrapper::CudaDevice_t) override {
    return "nvidia";
  }

  cuda_wrapper::CudaContext_t getPrimaryContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  cuda_wrapper::CudaContext_t createContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  void setContext(cuda_wrapper::CudaContext_t) override {}
  void releasePrimaryContext(cuda_wrapper::CudaDevice_t) override {}
  void releaseContext(cuda_wrapper::CudaContext_t) override {}
  cuda_wrapper::CudaStream_t createStream() override { return nullptr; }
  void destroyStream(cuda_wrapper::CudaStream_t) override {}
  cuda_wrapper::CudaEvent_t createEvent() override { return nullptr; }
  void destroyEvent(cuda_wrapper::CudaEvent_t) override {}
  cuda_wrapper::CudaModule_t loadModuleFromFile(const char *) override {
    return nullptr;
  }
  cuda_wrapper::CudaModule_t loadModuleFromImage(const void *) override {
    return nullptr;
  }
  cuda_wrapper::CudaFunction_t getFunction(cuda_wrapper::CudaModule_t,
                                           const char *) override {
    return nullptr;
  }
  void unloadModule(cuda_wrapper::CudaModule_t) override {}

private:
  cuda_wrapper::CudaContext_t context_{
      reinterpret_cast<cuda_wrapper::CudaContext_t>(0x1)};
};

cuda_api::CudaExecutionApi::ExecutionManager::Config makeCudaConfig(
    cuda_platform::CudaSlowOps *ops) {
  cuda_api::CudaExecutionApi::ExecutionManager::Config config{};
  config.slow_ops = ops;
  config.device_config.control_block_capacity = 1;
  config.device_config.control_block_block_size = 1;
  config.device_config.payload_capacity = 1;
  config.device_config.payload_block_size = 1;

  auto &context_cfg = config.device_config.context_config;
  context_cfg.control_block_capacity = 1;
  context_cfg.control_block_block_size = 1;
  context_cfg.payload_capacity = 1;
  context_cfg.payload_block_size = 1;

  config.kernel_base_config.control_block_capacity = 2;
  config.kernel_base_config.control_block_block_size = 2;
  config.kernel_base_config.payload_capacity = 2;
  config.kernel_base_config.payload_block_size = 2;

  config.kernel_metadata_config.control_block_capacity = 2;
  config.kernel_metadata_config.control_block_block_size = 2;
  config.kernel_metadata_config.payload_capacity = 2;
  config.kernel_metadata_config.payload_block_size = 2;
  return config;
}

void dummyCudaExecute(
    ::orteaf::internal::execution::cuda::resource::CudaKernelBase &,
    kernel::KernelArgs &) {}

class KernelMetadataCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    cuda_api::CudaExecutionApi::shutdown();
    cuda_api::CudaExecutionApi::configure(makeCudaConfig(new TestCudaSlowOps()));
  }

  void TearDown() override { cuda_api::CudaExecutionApi::shutdown(); }
};

TEST_F(KernelMetadataCudaTest, CudaMetadataRebuildsCudaEntry) {
  cuda_api::CudaExecutionApi::KernelKeys keys;
  keys.pushBack({cuda_api::CudaExecutionApi::ModuleKey::File("module.bin"),
                 "kernel_a"});

  auto metadata_lease = cuda_api::CudaExecutionApi::acquireKernelMetadata(keys);
  ASSERT_TRUE(metadata_lease);
  auto *metadata_ptr = metadata_lease.operator->();
  ASSERT_NE(metadata_ptr, nullptr);
  metadata_ptr->setExecute(dummyCudaExecute);

  kernel::core::KernelMetadataLease metadata{std::move(metadata_lease)};
  auto rebuilt_entry = metadata.rebuild();

  using CudaKernelBaseLease = kernel::core::KernelEntry::CudaKernelBaseLease;
  ASSERT_TRUE(std::holds_alternative<CudaKernelBaseLease>(rebuilt_entry.base()));
  auto *base_lease = std::get_if<CudaKernelBaseLease>(&rebuilt_entry.base());
  ASSERT_NE(base_lease, nullptr);
  ASSERT_TRUE(*base_lease);
  auto *base_ptr = base_lease->operator->();
  ASSERT_NE(base_ptr, nullptr);
  EXPECT_EQ(base_ptr->kernelCount(), 1u);
  EXPECT_EQ(base_ptr->execute(), dummyCudaExecute);
}

TEST_F(KernelMetadataCudaTest, FromEntryPreservesCudaKeysAndExecute) {
  cuda_api::CudaExecutionApi::KernelKeys keys;
  keys.pushBack({cuda_api::CudaExecutionApi::ModuleKey::File("module.bin"),
                 "kernel_a"});

  auto metadata_lease = cuda_api::CudaExecutionApi::acquireKernelMetadata(keys);
  ASSERT_TRUE(metadata_lease);
  auto *metadata_ptr = metadata_lease.operator->();
  ASSERT_NE(metadata_ptr, nullptr);
  metadata_ptr->setExecute(dummyCudaExecute);

  kernel::core::KernelMetadataLease metadata_from_cuda{
      std::move(metadata_lease)};
  auto entry = metadata_from_cuda.rebuild();
  auto metadata = kernel::core::KernelMetadataLease::fromEntry(entry);

  using CudaKernelMetadataLease =
      kernel::core::KernelMetadataLease::CudaKernelMetadataLease;
  ASSERT_TRUE(
      std::holds_alternative<CudaKernelMetadataLease>(metadata.lease()));
  auto *variant_lease = std::get_if<CudaKernelMetadataLease>(&metadata.lease());
  ASSERT_NE(variant_lease, nullptr);
  ASSERT_TRUE(*variant_lease);
  auto *rebuilt_metadata = variant_lease->operator->();
  ASSERT_NE(rebuilt_metadata, nullptr);
  ASSERT_EQ(rebuilt_metadata->keys().size(), 1u);
  EXPECT_EQ(rebuilt_metadata->keys()[0].first.identifier, "module.bin");
  EXPECT_EQ(rebuilt_metadata->keys()[0].second, "kernel_a");
  EXPECT_EQ(rebuilt_metadata->execute(), dummyCudaExecute);
}

#endif // ORTEAF_ENABLE_CUDA

} // namespace
