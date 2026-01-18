#include "orteaf/internal/runtime/cuda/manager/cuda_buffer_manager.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>

#if ORTEAF_ENABLE_CUDA

namespace cuda_rt = orteaf::internal::runtime::cuda::manager;
namespace cuda_platform = orteaf::internal::execution::cuda::platform;
namespace cuda_wrapper = orteaf::internal::execution::cuda::platform::wrapper;
namespace cuda_resource = orteaf::internal::execution::cuda::resource;

namespace {

std::size_t normalizeAlignment(std::size_t alignment) {
  if (alignment == 0) {
    return alignof(std::max_align_t);
  }
  if ((alignment & (alignment - 1)) != 0) {
    return alignof(std::max_align_t);
  }
  if (alignment < alignof(void *)) {
    return alignof(void *);
  }
  return alignment;
}

cuda_resource::CudaBufferView testAlloc(std::size_t size,
                                        std::size_t alignment) {
  if (size == 0) {
    return {};
  }
  const std::size_t aligned = normalizeAlignment(alignment);
  void *ptr = nullptr;
  if (posix_memalign(&ptr, aligned, size) != 0) {
    return {};
  }
  const auto raw = static_cast<cuda_wrapper::CudaDevicePtr_t>(
      reinterpret_cast<std::uintptr_t>(ptr));
  return cuda_resource::CudaBufferView{raw, 0, size};
}

void testFree(cuda_resource::CudaBufferView view, std::size_t,
              std::size_t) {
  if (!view) {
    return;
  }
  auto *ptr = reinterpret_cast<void *>(
      static_cast<std::uintptr_t>(view.raw()));
  std::free(ptr);
}

class TestCudaSlowOps final : public cuda_platform::CudaSlowOps {
public:
  int getDeviceCount() override { return 1; }

  cuda_wrapper::CudaDevice_t getDevice(std::uint32_t) override {
    return cuda_wrapper::CudaDevice_t{0};
  }

  cuda_wrapper::ComputeCapability getComputeCapability(
      cuda_wrapper::CudaDevice_t) override {
    return cuda_wrapper::ComputeCapability{0, 0};
  }

  std::string getDeviceName(cuda_wrapper::CudaDevice_t) override {
    return "mock-cuda";
  }

  std::string getDeviceVendor(cuda_wrapper::CudaDevice_t) override {
    return "mock";
  }

  cuda_wrapper::CudaContext_t getPrimaryContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  cuda_wrapper::CudaContext_t createContext(
      cuda_wrapper::CudaDevice_t) override {
    return context_;
  }

  void setContext(cuda_wrapper::CudaContext_t context) override {
    last_context_ = context;
  }

  void releasePrimaryContext(cuda_wrapper::CudaDevice_t) override {}

  void releaseContext(cuda_wrapper::CudaContext_t) override {}

  cuda_wrapper::CudaStream_t createStream() override { return nullptr; }

  void destroyStream(cuda_wrapper::CudaStream_t) override {}

  cuda_wrapper::CudaEvent_t createEvent() override { return nullptr; }

  void destroyEvent(cuda_wrapper::CudaEvent_t) override {}

  void setContextForTest(cuda_wrapper::CudaContext_t context) {
    context_ = context;
  }

  cuda_wrapper::CudaContext_t lastContext() const noexcept {
    return last_context_;
  }

private:
  cuda_wrapper::CudaContext_t context_{nullptr};
  cuda_wrapper::CudaContext_t last_context_{nullptr};
};

} // namespace

class CudaBufferManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    slow_ops_ = std::make_unique<TestCudaSlowOps>();
    manager_ = std::make_unique<cuda_rt::CudaBufferManager>();
    context_ = reinterpret_cast<cuda_wrapper::CudaContext_t>(0x1);
    slow_ops_->setContextForTest(context_);
  }

  void TearDown() override {
    manager_->shutdown();
    manager_.reset();
    slow_ops_.reset();
  }

  void configureManager() {
    cuda_rt::CudaBufferManager::Config config{};
    config.alloc = &testAlloc;
    config.free = &testFree;
    manager_->configureForTest(config, context_, slow_ops_.get());
  }

  std::unique_ptr<TestCudaSlowOps> slow_ops_;
  std::unique_ptr<cuda_rt::CudaBufferManager> manager_;
  cuda_wrapper::CudaContext_t context_{nullptr};
};

TEST_F(CudaBufferManagerTest, ConfigureSucceeds) {
  configureManager();
  EXPECT_TRUE(manager_->isConfiguredForTest());
}

TEST_F(CudaBufferManagerTest, ShutdownClearsState) {
  configureManager();
  manager_->shutdown();
  EXPECT_FALSE(manager_->isConfiguredForTest());
}

TEST_F(CudaBufferManagerTest, AcquireReturnsValidLease) {
  configureManager();

  auto lease = manager_->acquire(1024);
  EXPECT_TRUE(lease);
  auto *buffer = lease.operator->();
  ASSERT_NE(buffer, nullptr);
  EXPECT_TRUE(buffer->valid());
  EXPECT_NE(buffer->view.data(), 0u);
  EXPECT_EQ(buffer->view.size(), 1024u);
  EXPECT_EQ(slow_ops_->lastContext(), context_);
}

TEST_F(CudaBufferManagerTest, AcquireWithAlignmentSucceeds) {
  configureManager();

  constexpr std::size_t kAlignment = 64;
  auto lease = manager_->acquire(512, kAlignment);
  EXPECT_TRUE(lease);
  auto *buffer = lease.operator->();
  ASSERT_NE(buffer, nullptr);
  EXPECT_NE(buffer->view.data(), 0u);

  auto ptr = static_cast<std::uintptr_t>(buffer->view.data());
  EXPECT_EQ(ptr % kAlignment, 0u);
}

TEST_F(CudaBufferManagerTest, AcquireZeroSizeThrows) {
  configureManager();
  EXPECT_THROW(manager_->acquire(0), std::system_error);
}

TEST_F(CudaBufferManagerTest, MultipleAcquiresSucceed) {
  configureManager();

  auto lease1 = manager_->acquire(128);
  auto lease2 = manager_->acquire(256);
  auto lease3 = manager_->acquire(512);

  EXPECT_TRUE(lease1);
  EXPECT_TRUE(lease2);
  EXPECT_TRUE(lease3);

  auto *buffer1 = lease1.operator->();
  auto *buffer2 = lease2.operator->();
  auto *buffer3 = lease3.operator->();
  ASSERT_NE(buffer1, nullptr);
  ASSERT_NE(buffer2, nullptr);
  ASSERT_NE(buffer3, nullptr);
  EXPECT_NE(buffer1->view.data(), buffer2->view.data());
  EXPECT_NE(buffer2->view.data(), buffer3->view.data());
  EXPECT_NE(buffer1->view.data(), buffer3->view.data());
}

TEST_F(CudaBufferManagerTest, LeaseReleaseDecreasesRefCount) {
  configureManager();

  auto lease = manager_->acquire(1024);
  EXPECT_TRUE(lease);
  EXPECT_EQ(lease.strongCount(), 1u);

  auto lease_copy = lease;
  EXPECT_EQ(lease.strongCount(), 2u);
  EXPECT_EQ(lease_copy.strongCount(), 2u);

  lease_copy.release();
  EXPECT_EQ(lease.strongCount(), 1u);
}

TEST_F(CudaBufferManagerTest, BufferViewFromLeaseIsValid) {
  configureManager();

  auto lease = manager_->acquire(1024);
  auto *buffer = lease.operator->();
  ASSERT_NE(buffer, nullptr);

  EXPECT_TRUE(buffer->valid());
  EXPECT_EQ(buffer->view.size(), 1024u);
  EXPECT_EQ(buffer->view.offset(), 0u);
  EXPECT_EQ(buffer->view.data(), buffer->view.raw());
}

TEST_F(CudaBufferManagerTest, NotConfiguredThrows) {
  EXPECT_THROW(manager_->acquire(1024), std::system_error);
}

#endif // ORTEAF_ENABLE_CUDA
