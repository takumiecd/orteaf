#include <gmock/gmock.h>
#include <gtest/gtest.h>

#if ORTEAF_ENABLE_MPS

#include <orteaf/internal/execution/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/storage/mps/mps_storage.h>
#include <tests/internal/execution/mps/manager/testing/mock_mps_resource.h>

namespace mps_storage = orteaf::internal::storage::mps;
namespace mps = orteaf::internal::execution::mps;
namespace mps_rt = orteaf::internal::execution::mps::manager;

namespace {

// =============================================================================
// MpsStorage Builder Tests (Using StubMpsResource - No GPU required)
// =============================================================================

class MpsStorageBuilderTest : public ::testing::Test {
protected:
  using StubResource = orteaf::tests::execution::mps::StubMpsResource;
  using Manager = mps_rt::MpsBufferManager<StubResource>;
  using Config = typename Manager::Config;
  using LaunchParams = typename Manager::LaunchParams;

  void SetUp() override {
    Config cfg{};
    cfg.min_block_size = 64;
    cfg.max_block_size = 1024 * 1024;
    cfg.chunk_size = 1024 * 1024;
    cfg.payload_capacity = 16;
    cfg.control_block_capacity = 16;
    cfg.payload_block_size = 1;
    cfg.control_block_block_size = 1;
    manager_.configureForTest(cfg, device(), mps::MpsDeviceHandle{0}, heap(),
                              nullptr);
  }

  void TearDown() override { manager_.shutdown(); }

  // Dummy device for StubMpsResource
  typename Manager::DeviceType device() {
    return reinterpret_cast<typename Manager::DeviceType>(0x12345678);
  }

  void *heap() { return reinterpret_cast<void *>(0x87654321); }

  Manager &manager() { return manager_; }

  Manager manager_{};
};

// ============================================================
// Builder tests
// ============================================================

TEST_F(MpsStorageBuilderTest, BuilderCreatesValidStorage) {
  // Note: MpsStorage uses the concrete MpsResource type, but we can test
  // the builder pattern with a custom manager type if needed.
  // For now, we just test that the storage can be default constructed.
  mps_storage::MpsStorage storage;
  SUCCEED();
}

TEST_F(MpsStorageBuilderTest, DefaultConstructedStorageExists) {
  mps_storage::MpsStorage storage;
  // Default constructed should be valid (just empty)
  SUCCEED();
}

TEST_F(MpsStorageBuilderTest, StorageIsMoveConstructible) {
  mps_storage::MpsStorage storage1;
  mps_storage::MpsStorage storage2 = std::move(storage1);
  SUCCEED();
}

TEST_F(MpsStorageBuilderTest, StorageIsMoveAssignable) {
  mps_storage::MpsStorage storage1;
  mps_storage::MpsStorage storage2;
  storage2 = std::move(storage1);
  SUCCEED();
}

TEST_F(MpsStorageBuilderTest, StorageIsCopyConstructible) {
  mps_storage::MpsStorage storage1;
  mps_storage::MpsStorage storage2 = storage1;
  SUCCEED();
}

TEST_F(MpsStorageBuilderTest, StorageIsCopyAssignable) {
  mps_storage::MpsStorage storage1;
  mps_storage::MpsStorage storage2;
  storage2 = storage1;
  SUCCEED();
}

} // namespace

#endif // ORTEAF_ENABLE_MPS
