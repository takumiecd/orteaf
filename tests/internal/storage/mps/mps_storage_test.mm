#include <gmock/gmock.h>
#include <gtest/gtest.h>

#if ORTEAF_ENABLE_MPS

#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/execution/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_fence_manager.h>
#include <orteaf/internal/execution/mps/mps_handles.h>
#include <orteaf/internal/execution/mps/platform/mps_slow_ops.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_command_queue.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#include <orteaf/internal/storage/mps/mps_storage.h>
#include <tests/internal/execution/mps/manager/testing/execution_mock_expectations.h>
#include <tests/internal/execution/mps/manager/testing/mock_mps_resource.h>

namespace mps_storage = orteaf::internal::storage::mps;
namespace mps = orteaf::internal::execution::mps;
namespace mps_api = orteaf::internal::execution::mps::api;
namespace mps_rt = orteaf::internal::execution::mps::manager;
namespace mps_wrapper = orteaf::internal::execution::mps::platform::wrapper;
namespace testing_mps = orteaf::tests::execution::mps;

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

// ============================================================
// Buffer view accessor tests
// ============================================================

TEST_F(MpsStorageBuilderTest, DefaultStorageHasEmptyBufferView) {
  mps_storage::MpsStorage storage;
  auto view = storage.bufferView();
  EXPECT_FALSE(view);
}

TEST_F(MpsStorageBuilderTest, DefaultStorageHasNullBuffer) {
  mps_storage::MpsStorage storage;
  EXPECT_EQ(storage.buffer(), nullptr);
}

TEST_F(MpsStorageBuilderTest, DefaultStorageHasZeroOffset) {
  mps_storage::MpsStorage storage;
  EXPECT_EQ(storage.bufferOffset(), 0u);
}

// =============================================================================
// withFenceToken Tests
// =============================================================================

class MpsStorageFenceTokenTest : public ::testing::Test {
protected:
  using FenceManager = mps_rt::MpsFenceManager;
  using FenceToken = mps_storage::MpsStorage::FenceToken;
  using SlowOps = mps::platform::MpsSlowOps;

  void SetUp() override {
    device_ = mps::platform::wrapper::getDevice();
    if (device_ == nullptr) {
      GTEST_SKIP() << "No Metal devices available";
    }
    queue_ = mps::platform::wrapper::createCommandQueue(device_);
    if (queue_ == nullptr) {
      GTEST_SKIP() << "Failed to create command queue";
    }

    FenceManager::Config fence_cfg{};
    fence_cfg.control_block_capacity = 4;
    fence_cfg.control_block_block_size = 2;
    fence_cfg.payload_capacity = 4;
    fence_cfg.payload_block_size = 2;
    fence_manager_.configureForTest(fence_cfg, device_, &ops_);
  }

  void TearDown() override {
    fence_manager_.shutdown();
    if (queue_ != nullptr) {
      mps::platform::wrapper::destroyCommandQueue(queue_);
      queue_ = nullptr;
    }
    if (device_ != nullptr) {
      mps::platform::wrapper::deviceRelease(device_);
      device_ = nullptr;
    }
  }

  FenceToken createFenceToken() {
    FenceToken token;
    auto lease = fence_manager_.acquire();
    auto *payload = lease.operator->();
    if (payload != nullptr) {
      payload->setCommandQueueHandle(mps::MpsCommandQueueHandle{1});
    }
    token.setWriteFence(std::move(lease));
    return token;
  }

  mps::platform::wrapper::MpsDevice_t device_{nullptr};
  mps::platform::wrapper::MpsCommandQueue_t queue_{nullptr};
  FenceManager fence_manager_{};
  mps::platform::MpsSlowOpsImpl ops_{};
};

TEST_F(MpsStorageFenceTokenTest, BuilderWithFenceTokenCreatesStorage) {
  auto token = createFenceToken();
  EXPECT_TRUE(token.hasWriteFence());

  // Builder should accept the fence token
  auto builder =
      mps_storage::MpsStorage::builder().withFenceToken(std::move(token));
  SUCCEED();
}

TEST_F(MpsStorageFenceTokenTest, FenceTokenCopyPreservesLeases) {
  auto token = createFenceToken();
  EXPECT_TRUE(token.hasWriteFence());

  FenceToken copied = token;
  EXPECT_TRUE(copied.hasWriteFence());
  EXPECT_TRUE(token.hasWriteFence());
}

TEST_F(MpsStorageFenceTokenTest, FenceTokenMoveTransfersLeases) {
  auto token = createFenceToken();
  EXPECT_TRUE(token.hasWriteFence());

  FenceToken moved = std::move(token);
  EXPECT_TRUE(moved.hasWriteFence());
  EXPECT_TRUE(token.empty());
}

TEST(MpsStorageExecutionApiTest, BuilderWithDeviceHandleUsesExecutionApi) {
  using MockOps = testing_mps::MpsExecutionOpsMock;
  auto *mock = new ::testing::NiceMock<MockOps>();

  const auto device = reinterpret_cast<mps_wrapper::MpsDevice_t>(0xD1);
  const auto command_queue =
      reinterpret_cast<mps_wrapper::MpsCommandQueue_t>(0xC1);
  const auto event = reinterpret_cast<mps_wrapper::MpsEvent_t>(0xE1);
  const auto fence = reinterpret_cast<mps_wrapper::MpsFence_t>(0xF1);
  const auto heap_descriptor =
      reinterpret_cast<mps_wrapper::MpsHeapDescriptor_t>(0xA1);
  const auto heap = reinterpret_cast<mps_wrapper::MpsHeap_t>(0xB1);

  testing_mps::ExecutionMockExpectations::expectGetDeviceCount(*mock, 1);
  testing_mps::ExecutionMockExpectations::expectGetDevices(*mock,
                                                           {{0, device}});
  testing_mps::ExecutionMockExpectations::expectDetectArchitectures(
      *mock, {{mps::MpsDeviceHandle{0},
               ::orteaf::internal::architecture::Architecture::MpsGeneric}});
  testing_mps::ExecutionMockExpectations::expectCreateCommandQueues(
      *mock, {command_queue}, ::testing::Eq(device));
  testing_mps::ExecutionMockExpectations::expectCreateEvents(
      *mock, {event}, ::testing::Eq(device));
  testing_mps::ExecutionMockExpectations::expectCreateFences(
      *mock, {fence}, ::testing::Eq(device));

  testing_mps::ExecutionMockExpectations::expectCreateHeapDescriptors(
      *mock, {heap_descriptor});
  testing_mps::ExecutionMockExpectations::expectCreateHeaps(
      *mock, {heap}, ::testing::Eq(device), ::testing::Eq(heap_descriptor));
  testing_mps::ExecutionMockExpectations::expectDestroyHeapDescriptors(
      *mock, {heap_descriptor});
  testing_mps::ExecutionMockExpectations::expectDestroyHeaps(*mock, {heap});

  testing_mps::ExecutionMockExpectations::expectDestroyCommandQueues(
      *mock, {command_queue});
  testing_mps::ExecutionMockExpectations::expectDestroyEvents(*mock, {event});
  testing_mps::ExecutionMockExpectations::expectDestroyFences(*mock, {fence});
  testing_mps::ExecutionMockExpectations::expectReleaseDevices(*mock, {device});

  mps_rt::HeapDescriptorKey heap_key = mps_rt::HeapDescriptorKey::Sized(256);
  EXPECT_CALL(*mock,
              setHeapDescriptorSize(heap_descriptor, heap_key.size_bytes))
      .Times(1);
  EXPECT_CALL(*mock, setHeapDescriptorResourceOptions(
                         heap_descriptor, heap_key.resource_options))
      .Times(1);
  EXPECT_CALL(*mock, setHeapDescriptorStorageMode(heap_descriptor,
                                                  heap_key.storage_mode))
      .Times(1);
  EXPECT_CALL(*mock, setHeapDescriptorCPUCacheMode(heap_descriptor,
                                                   heap_key.cpu_cache_mode))
      .Times(1);
  EXPECT_CALL(*mock, setHeapDescriptorHazardTrackingMode(
                         heap_descriptor, heap_key.hazard_tracking_mode))
      .Times(1);
  EXPECT_CALL(*mock, setHeapDescriptorType(heap_descriptor, heap_key.heap_type))
      .Times(1);

  mps_api::MpsExecutionApi::ExecutionManager::Config config{};
  config.slow_ops = mock;
  auto &device_config = config.device_config;
  device_config.control_block_capacity = 1;
  device_config.control_block_block_size = 1;
  device_config.control_block_growth_chunk_size = 1;
  device_config.payload_capacity = 1;
  device_config.payload_block_size = 1;
  device_config.payload_growth_chunk_size = 1;

  auto &queue_config = device_config.command_queue_config;
  queue_config.control_block_capacity = 1;
  queue_config.control_block_block_size = 1;
  queue_config.control_block_growth_chunk_size = 1;
  queue_config.payload_capacity = 1;
  queue_config.payload_block_size = 1;
  queue_config.payload_growth_chunk_size = 1;

  auto &event_config = device_config.event_config;
  event_config.control_block_capacity = 1;
  event_config.control_block_block_size = 1;
  event_config.control_block_growth_chunk_size = 1;
  event_config.payload_capacity = 1;
  event_config.payload_block_size = 1;
  event_config.payload_growth_chunk_size = 1;

  auto &fence_config = device_config.fence_config;
  fence_config.control_block_capacity = 1;
  fence_config.control_block_block_size = 1;
  fence_config.control_block_growth_chunk_size = 1;
  fence_config.payload_capacity = 1;
  fence_config.payload_block_size = 1;
  fence_config.payload_growth_chunk_size = 1;

  auto &heap_config = device_config.heap_config;
  heap_config.control_block_capacity = 1;
  heap_config.control_block_block_size = 1;
  heap_config.control_block_growth_chunk_size = 1;
  heap_config.payload_capacity = 1;
  heap_config.payload_block_size = 1;
  heap_config.payload_growth_chunk_size = 1;

  auto &buffer_config = heap_config.buffer_config;
  buffer_config.control_block_capacity = 1;
  buffer_config.control_block_block_size = 1;
  buffer_config.control_block_growth_chunk_size = 1;
  buffer_config.payload_capacity = 1;
  buffer_config.payload_block_size = 1;
  buffer_config.payload_growth_chunk_size = 1;

  auto &library_config = device_config.library_config;
  library_config.control_block_capacity = 1;
  library_config.control_block_block_size = 1;
  library_config.control_block_growth_chunk_size = 1;
  library_config.payload_capacity = 1;
  library_config.payload_block_size = 1;
  library_config.payload_growth_chunk_size = 1;

  auto &pipeline_config = library_config.pipeline_config;
  pipeline_config.control_block_capacity = 1;
  pipeline_config.control_block_block_size = 1;
  pipeline_config.control_block_growth_chunk_size = 1;
  pipeline_config.payload_capacity = 1;
  pipeline_config.payload_block_size = 1;
  pipeline_config.payload_growth_chunk_size = 1;

  auto &graph_config = device_config.graph_config;
  graph_config.control_block_capacity = 1;
  graph_config.control_block_block_size = 1;
  graph_config.control_block_growth_chunk_size = 1;
  graph_config.payload_capacity = 1;
  graph_config.payload_block_size = 1;
  graph_config.payload_growth_chunk_size = 1;

  mps_api::MpsExecutionApi::configure(config);
  {
    auto storage = mps_storage::MpsStorage::builder()
                       .withDeviceHandle(mps::MpsDeviceHandle{0}, heap_key)
                       .withNumElements(0)
                       .build();
    SUCCEED();
  }
  mps_api::MpsExecutionApi::shutdown();
}

} // namespace

#endif // ORTEAF_ENABLE_MPS
