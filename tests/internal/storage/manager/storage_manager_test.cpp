#include <gtest/gtest.h>

#include <system_error>

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/storage/manager/storage_manager.h"
#include "orteaf/internal/storage/storage.h"
#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_device.h"
#include "orteaf/internal/storage/mps/mps_storage.h"
#endif

namespace storage = orteaf::internal::storage;
namespace storage_manager = orteaf::internal::storage::manager;
namespace cpu_api = orteaf::internal::execution::cpu::api;
namespace cpu = orteaf::internal::execution::cpu;
#if ORTEAF_ENABLE_MPS
namespace mps_api = orteaf::internal::execution::mps::api;
namespace mps_platform = orteaf::internal::execution::mps::platform::wrapper;
#endif

class StorageManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
    cpu_api::CpuExecutionApi::configure(config);
    manager_.configure({});
  }

  void TearDown() override {
    manager_.shutdown();
    cpu_api::CpuExecutionApi::shutdown();
  }

  storage_manager::StorageManager manager_{};
};

TEST_F(StorageManagerTest, AcquireCpuStorageFromUnifiedManager) {
  storage_manager::CpuStorageRequest request{};
  request.device = cpu::CpuDeviceHandle{0};
  request.size = 256;
  request.alignment = 16;

  auto lease = manager_.acquire(storage_manager::StorageRequest{request});
  ASSERT_TRUE(lease);

  auto *payload = lease.operator->();
  ASSERT_NE(payload, nullptr);
  EXPECT_TRUE(payload->valid());
  EXPECT_NE(payload->tryAs<storage::CpuStorage>(), nullptr);
}

TEST_F(StorageManagerTest, AcquireCpuStorageFromManagerRequestAlias) {
  storage_manager::CpuStorageRequest request{};
  request.device = cpu::CpuDeviceHandle{0};
  request.size = 128;

  storage_manager::StorageManager::Request request_variant{request};
  auto lease = manager_.acquire(request_variant);
  ASSERT_TRUE(lease);

  auto *payload = lease.operator->();
  ASSERT_NE(payload, nullptr);
  EXPECT_TRUE(payload->valid());
  EXPECT_NE(payload->tryAs<storage::CpuStorage>(), nullptr);
}

TEST_F(StorageManagerTest, InvalidRequestThrows) {
  storage_manager::CpuStorageRequest request{};
  request.device = cpu::CpuDeviceHandle{0};
  request.size = 0;

  EXPECT_THROW(manager_.acquire(storage_manager::StorageRequest{request}),
               std::system_error);
}

TEST(StorageManagerStandaloneTest, AcquireWithoutConfigureThrows) {
  storage_manager::StorageManager manager{};
  storage_manager::CpuStorageRequest request{};
  request.device = cpu::CpuDeviceHandle{0};
  request.size = 64;

  EXPECT_THROW(manager.acquire(storage_manager::StorageRequest{request}),
               std::system_error);
}

#if ORTEAF_ENABLE_MPS
namespace {

template <typename ConfigT>
void configurePoolConfig(ConfigT &config, std::size_t capacity) {
  config.control_block_capacity = capacity;
  config.control_block_block_size = 1;
  config.control_block_growth_chunk_size = 1;
  config.payload_capacity = capacity;
  config.payload_block_size = 1;
  config.payload_growth_chunk_size = 1;
}

void configureMpsExecutionApi(std::size_t device_count,
                              std::size_t heap_size_bytes) {
  mps_api::MpsExecutionApi::ExecutionManager::Config config{};
  auto &device_config = config.device_config;
  configurePoolConfig(device_config, device_count);
  configurePoolConfig(device_config.command_queue_config, 1);
  configurePoolConfig(device_config.event_config, 1);
  configurePoolConfig(device_config.fence_config, 1);
  configurePoolConfig(device_config.heap_config, 1);
  configurePoolConfig(device_config.heap_config.buffer_config, 1);
  device_config.heap_config.buffer_config.chunk_size = heap_size_bytes / 4;
  device_config.heap_config.buffer_config.max_block_size =
      device_config.heap_config.buffer_config.chunk_size;
  device_config.heap_config.buffer_config.min_block_size = 64;
  configurePoolConfig(device_config.library_config, 1);
  configurePoolConfig(device_config.library_config.pipeline_config, 1);
  configurePoolConfig(device_config.graph_config, 1);
  mps_api::MpsExecutionApi::configure(config);
}

} // namespace

TEST(StorageManagerMpsTest, AcquireMpsStorageFromUnifiedManager) {
  const int device_count = mps_platform::getDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "No MPS devices available";
  }

  constexpr std::size_t kHeapSizeBytes = 1024 * 1024;
  storage_manager::StorageManager manager{};
  configureMpsExecutionApi(static_cast<std::size_t>(device_count),
                           kHeapSizeBytes);
  manager.configure({});

  storage_manager::MpsStorageRequest request{};
  request.device = ::orteaf::internal::execution::mps::MpsDeviceHandle{0};
  request.heap_key =
      storage_manager::MpsStorageRequest::HeapDescriptorKey::Sized(
          kHeapSizeBytes);
  request.size = 256;
  request.alignment = 16;

  {
    auto lease = manager.acquire(storage_manager::StorageRequest{request});
    ASSERT_TRUE(lease);

    auto *payload = lease.operator->();
    ASSERT_NE(payload, nullptr);
    EXPECT_TRUE(payload->valid());
    EXPECT_NE(payload->tryAs<storage::MpsStorage>(), nullptr);
  }

  manager.shutdown();
  mps_api::MpsExecutionApi::shutdown();
}
#endif // ORTEAF_ENABLE_MPS
