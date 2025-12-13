#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/runtime/mps/manager/mps_buffer_manager.h>
#include <tests/internal/testing/error_assert.h>

namespace base = orteaf::internal::base;
namespace diag_error = orteaf::internal::diagnostics::error;
namespace mps_rt = orteaf::internal::runtime::mps::manager;

using orteaf::tests::ExpectError;

namespace {

#if ORTEAF_ENABLE_MPS

// ============================================================================
// Phase 1: シンプルなテスト（GPUリソース不要）
// ============================================================================

class MpsBufferManagerSimpleTest : public ::testing::Test {
protected:
  using Manager = mps_rt::MpsBufferManager;

  Manager &manager() { return manager_; }

  Manager manager_{};
};

// --- Shutdown Tests (未初期化時) ---

TEST_F(MpsBufferManagerSimpleTest, ShutdownWithoutInitializeIsNoOp) {
  EXPECT_NO_THROW(manager().shutdown());
}

TEST_F(MpsBufferManagerSimpleTest,
       MultipleShutdownsWithoutInitializeAreIdempotent) {
  EXPECT_NO_THROW(manager().shutdown());
  EXPECT_NO_THROW(manager().shutdown());
  EXPECT_NO_THROW(manager().shutdown());
}

// --- State Tests (未初期化時) ---

TEST_F(MpsBufferManagerSimpleTest, IsNotInitializedByDefault) {
  EXPECT_FALSE(manager().isInitialized());
}

TEST_F(MpsBufferManagerSimpleTest, CapacityIsZeroBeforeInitialize) {
  EXPECT_EQ(manager().capacity(), 0u);
}

TEST_F(MpsBufferManagerSimpleTest, AcquireBeforeInitializationThrows) {
  ExpectError(diag_error::OrteafErrc::InvalidState,
              [&] { (void)manager().acquire(1024, 16); });
}

TEST_F(MpsBufferManagerSimpleTest, AcquireByHandleBeforeInitializationThrows) {
  base::BufferHandle handle{0, 1};
  ExpectError(diag_error::OrteafErrc::InvalidState,
              [&] { (void)manager().acquire(handle); });
}

// --- GrowthChunkSize Tests ---

TEST_F(MpsBufferManagerSimpleTest, DefaultGrowthChunkSizeIsOne) {
  EXPECT_EQ(manager().growthChunkSize(), 1u);
}

TEST_F(MpsBufferManagerSimpleTest, SetGrowthChunkSizeWorks) {
  manager().setGrowthChunkSize(10);
  EXPECT_EQ(manager().growthChunkSize(), 10u);
}

TEST_F(MpsBufferManagerSimpleTest, SetGrowthChunkSizeToZeroThrows) {
  ExpectError(diag_error::OrteafErrc::InvalidArgument,
              [&] { manager().setGrowthChunkSize(0); });
}

#endif // ORTEAF_ENABLE_MPS

} // namespace

// ============================================================================
// Phase 2: GPUリソースを使った統合テスト
// ============================================================================
#if ORTEAF_ENABLE_MPS

#include <orteaf/internal/runtime/mps/manager/mps_library_manager.h>
#include <orteaf/internal/runtime/mps/platform/mps_slow_ops.h>
#include <orteaf/internal/runtime/mps/platform/wrapper/mps_heap.h>

namespace mps_platform = orteaf::internal::runtime::mps::platform;
namespace mps_wrapper = orteaf::internal::runtime::mps::platform::wrapper;

namespace {

class MpsBufferManagerIntegrationTest : public ::testing::Test {
protected:
  using Manager = mps_rt::MpsBufferManager;
  using Config = typename Manager::Config;

  void SetUp() override {
    // Acquire real MPS device
    const int count = ops_.getDeviceCount();
    if (count <= 0) {
      GTEST_SKIP() << "No MPS devices available";
    }
    device_ = ops_.getDevice(0);
    if (device_ == nullptr) {
      GTEST_SKIP() << "Unable to acquire MPS device";
    }

    // Create heap descriptor
    heap_descriptor_ = mps_wrapper::createHeapDescriptor();
    if (!heap_descriptor_) {
      GTEST_SKIP() << "Failed to create heap descriptor";
    }
    mps_wrapper::setHeapDescriptorSize(heap_descriptor_,
                                       64 * 1024 * 1024); // 64MB

    // Create heap
    heap_ = mps_wrapper::createHeap(device_, heap_descriptor_);
    if (!heap_) {
      mps_wrapper::destroyHeapDescriptor(heap_descriptor_);
      GTEST_SKIP() << "Failed to create heap";
    }

    // Initialize library manager (required by MpsResource)
    lib_manager_.initialize(device_, &ops_, 16);

    setup_successful_ = true;
  }

  void TearDown() override {
    manager_.shutdown();
    lib_manager_.shutdown();
    if (heap_) {
      mps_wrapper::destroyHeap(heap_);
      heap_ = nullptr;
    }
    if (heap_descriptor_) {
      mps_wrapper::destroyHeapDescriptor(heap_descriptor_);
      heap_descriptor_ = nullptr;
    }
  }

  Manager &manager() { return manager_; }

  void initializeManager(std::size_t capacity = 8) {
    Config cfg{};
    cfg.min_block_size = 64;
    cfg.max_block_size = 16 * 1024 * 1024;
    cfg.chunk_size = 16 * 1024 * 1024;
    manager_.initialize(device_, base::DeviceHandle{0}, heap_, &lib_manager_,
                        cfg, capacity);
  }

  mps_wrapper::MPSDevice_t device() { return device_; }
  bool setupSuccessful() const { return setup_successful_; }

  mps_platform::MpsSlowOpsImpl ops_{};
  Manager manager_{};
  mps_wrapper::MPSDevice_t device_{nullptr};
  mps_wrapper::MPSHeapDescriptor_t heap_descriptor_{nullptr};
  mps_wrapper::MPSHeap_t heap_{nullptr};
  mps_rt::MpsLibraryManager lib_manager_{};
  bool setup_successful_{false};
};

// --- Initialize Tests ---

TEST_F(MpsBufferManagerIntegrationTest, InitializeSucceeds) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  EXPECT_NO_THROW(initializeManager());
  EXPECT_TRUE(manager().isInitialized());
}

TEST_F(MpsBufferManagerIntegrationTest, InitializeWithZeroCapacitySucceeds) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  EXPECT_NO_THROW(initializeManager(0));
  EXPECT_TRUE(manager().isInitialized());
  EXPECT_EQ(manager().capacity(), 0u);
}

TEST_F(MpsBufferManagerIntegrationTest, ShutdownAfterInitializeWorks) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  initializeManager();
  EXPECT_NO_THROW(manager().shutdown());
  EXPECT_FALSE(manager().isInitialized());
}

TEST_F(MpsBufferManagerIntegrationTest, MultipleInitializeShutdownCyclesWork) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_THROW(initializeManager());
    EXPECT_TRUE(manager().isInitialized());
    EXPECT_NO_THROW(manager().shutdown());
    EXPECT_FALSE(manager().isInitialized());
  }
}

// --- Acquire/Release Tests ---

TEST_F(MpsBufferManagerIntegrationTest, AcquireReturnsValidLease) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  initializeManager();

  auto lease = manager().acquire(1024, 16);
  EXPECT_TRUE(lease);
  EXPECT_TRUE(lease.handle().isValid());

  manager().release(lease);
}

TEST_F(MpsBufferManagerIntegrationTest,
       AcquireWithZeroSizeReturnsInvalidLease) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  initializeManager();

  auto lease = manager().acquire(0, 16);
  EXPECT_FALSE(lease);
}

TEST_F(MpsBufferManagerIntegrationTest, MultipleAllocationsWork) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  initializeManager();

  auto lease1 = manager().acquire(1024, 16);
  auto lease2 = manager().acquire(2048, 32);
  auto lease3 = manager().acquire(4096, 64);

  EXPECT_TRUE(lease1);
  EXPECT_TRUE(lease2);
  EXPECT_TRUE(lease3);
  EXPECT_NE(lease1.handle().index, lease2.handle().index);
  EXPECT_NE(lease2.handle().index, lease3.handle().index);
  EXPECT_NE(lease1.handle().index, lease3.handle().index);

  manager().release(lease1);
  manager().release(lease2);
  manager().release(lease3);
}

TEST_F(MpsBufferManagerIntegrationTest, BufferRecyclingReusesSlots) {
  if (!setupSuccessful()) {
    GTEST_SKIP() << "GPU setup failed";
  }
  initializeManager();

  auto first = manager().acquire(1024, 16);
  const auto first_index = first.handle().index;
  manager().release(first);

  // Should reuse the same slot
  auto second = manager().acquire(2048, 16);
  EXPECT_EQ(second.handle().index, first_index);

  manager().release(second);
}

} // namespace

#endif // ORTEAF_ENABLE_MPS

// ============================================================================
// Phase 3: モックテスト（StubMpsResource を使用、GPU 不要）
// ============================================================================
#if ORTEAF_ENABLE_MPS

#include <tests/internal/runtime/mps/manager/testing/mock_mps_resource.h>

// StubMpsResource を使った MpsBufferManager
using StubBufferManager =
    mps_rt::MpsBufferManagerT<orteaf::tests::runtime::mps::StubMpsResource>;

class MpsBufferManagerMockTest : public ::testing::Test {
protected:
  using Manager = StubBufferManager;
  using Config = typename Manager::Config;
  using StubResource = orteaf::tests::runtime::mps::StubMpsResource;

  Manager &manager() { return manager_; }

  // StubMpsResource 用のダミーデバイス
  typename Manager::DeviceType device() {
    return reinterpret_cast<typename Manager::DeviceType>(0x12345678);
  }

  void *heap() { return reinterpret_cast<void *>(0x87654321); }

  void initializeManager(std::size_t capacity = 8) {
    Config cfg{};
    cfg.min_block_size = 64;
    cfg.max_block_size = 1024 * 1024;
    cfg.chunk_size = 1024 * 1024;
    manager_.initialize(device(), base::DeviceHandle{0}, heap(), nullptr, cfg,
                        capacity);
  }

  Manager manager_{};
};

// --- Initialize Tests (モック) ---

TEST_F(MpsBufferManagerMockTest, InitializeSucceeds) {
  EXPECT_NO_THROW(initializeManager());
  EXPECT_TRUE(manager_.isInitialized());
}

TEST_F(MpsBufferManagerMockTest, ShutdownAfterInitializeWorks) {
  initializeManager();
  EXPECT_NO_THROW(manager_.shutdown());
  EXPECT_FALSE(manager_.isInitialized());
}

TEST_F(MpsBufferManagerMockTest, AcquireReturnsValidLease) {
  initializeManager();

  auto lease = manager_.acquire(1024, 16);
  EXPECT_TRUE(lease);
  EXPECT_TRUE(lease.handle().isValid());

  manager_.release(lease);
}

TEST_F(MpsBufferManagerMockTest, MultipleAcquisitionsWork) {
  initializeManager();

  auto lease1 = manager_.acquire(256, 16);
  auto lease2 = manager_.acquire(512, 32);
  auto lease3 = manager_.acquire(1024, 64);

  EXPECT_TRUE(lease1);
  EXPECT_TRUE(lease2);
  EXPECT_TRUE(lease3);
  EXPECT_NE(lease1.handle().index, lease2.handle().index);
  EXPECT_NE(lease2.handle().index, lease3.handle().index);

  manager_.release(lease1);
  manager_.release(lease2);
  manager_.release(lease3);
}

TEST_F(MpsBufferManagerMockTest, ReleaseRecyclesSlot) {
  initializeManager();

  auto first = manager_.acquire(256, 16);
  const auto first_index = first.handle().index;
  manager_.release(first);

  // Should reuse the same slot
  auto second = manager_.acquire(512, 16);
  EXPECT_EQ(second.handle().index, first_index);

  manager_.release(second);
}

TEST_F(MpsBufferManagerMockTest, AcquireByHandleIncreasesRefCount) {
  initializeManager();

  auto lease1 = manager_.acquire(256, 16);
  const auto handle = lease1.handle();

  // Acquire by handle
  auto lease2 = manager_.acquire(handle);
  EXPECT_TRUE(lease2);
  EXPECT_EQ(lease2.handle().index, handle.index);
  EXPECT_EQ(lease2.handle().generation, handle.generation);

  // Release both
  manager_.release(lease1);
  manager_.release(lease2);
}

TEST_F(MpsBufferManagerMockTest, AcquireWithZeroSizeReturnsInvalidLease) {
  initializeManager();

  auto lease = manager_.acquire(0, 16);
  EXPECT_FALSE(lease);
}

TEST_F(MpsBufferManagerMockTest, CapacityGrowsOnAcquire) {
  initializeManager(4);

  // Acquire to use capacity
  auto lease1 = manager_.acquire(256, 16);
  auto lease2 = manager_.acquire(256, 16);
  EXPECT_GE(manager_.capacity(), 2u);

  manager_.release(lease1);
  manager_.release(lease2);
}

#endif // ORTEAF_ENABLE_MPS
