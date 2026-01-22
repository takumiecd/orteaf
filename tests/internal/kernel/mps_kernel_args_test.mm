#include "orteaf/internal/kernel/access.h"
#include "orteaf/internal/kernel/kernel_args.h"
#include "orteaf/internal/kernel/kernel_key.h"
#include "orteaf/internal/kernel/mps/mps_kernel_args.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution_context/mps/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;
namespace kk = kernel::kernel_key;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

// ============================================================
// Test Params struct (POD)
// ============================================================

struct MpsTestParams {
  static constexpr std::uint64_t kTypeId = 2;
  float alpha{0.0f};
  float beta{0.0f};
  int n{0};
};
static_assert(std::is_trivially_copyable_v<MpsTestParams>);

// OtherMpsParams for type mismatch testing
struct OtherMpsParams {
  static constexpr std::uint64_t kTypeId = 888;
  float x{0.0f};
};
static_assert(std::is_trivially_copyable_v<OtherMpsParams>);

// ============================================================
// MpsKernelArgs tests
// ============================================================

using MpsArgs = kernel::mps::MpsKernelArgs;

TEST(MpsKernelArgs, HostDefaultConstruct) {
  MpsArgs host;
  EXPECT_EQ(host.storageCount(), 0);
}

TEST(MpsKernelArgs, SetAndGetParams) {
  MpsArgs args;
  MpsTestParams params{1.0f, 2.0f, 100};

  auto key = kk::make(static_cast<Op>(1), Execution::Mps,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_TRUE(args.setParams(params, key));
  EXPECT_EQ(args.paramsSize(), sizeof(MpsTestParams));
  EXPECT_EQ(args.kernelKey(), key);

  MpsTestParams retrieved{};
  EXPECT_TRUE(args.getParams(retrieved, key));
  EXPECT_FLOAT_EQ(retrieved.alpha, 1.0f);
  EXPECT_FLOAT_EQ(retrieved.beta, 2.0f);
  EXPECT_EQ(retrieved.n, 100);
}

TEST(MpsKernelArgs, StorageManagement) {
  MpsArgs args;
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_EQ(args.storageCapacity(), MpsArgs::kMaxBindings);

  // Test clearing
  args.clearStorages();
  EXPECT_EQ(args.storageCount(), 0);
}

TEST(MpsKernelArgs, AddStorageLease) {
  MpsArgs args;

  // Add a storage lease (using default-constructed lease for now)
  kernel::mps::MpsKernelArgs::StorageLease lease;
  args.addStorageLease(std::move(lease), kernel::Access::Read);

  EXPECT_EQ(args.storageCount(), 1);
  EXPECT_EQ(args.storageAccessAt(0), kernel::Access::Read);
}

TEST(MpsKernelArgs, ParamsRawAccessors) {
  MpsArgs args;
  MpsTestParams params{3.14f, 2.71f, 42};

  auto key = kk::make(static_cast<Op>(1), Execution::Mps,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_TRUE(args.setParams(params, key));

  // Test raw data access
  const std::byte *data = args.paramsData();
  EXPECT_NE(data, nullptr);
  EXPECT_EQ(args.paramsSize(), sizeof(MpsTestParams));
  EXPECT_EQ(args.paramsCapacity(), MpsArgs::kParamBytes);

  // Test non-const data access
  std::byte *mutable_data = args.paramsData();
  EXPECT_NE(mutable_data, nullptr);
}

TEST(MpsKernelArgs, SetParamsRaw) {
  MpsArgs args;
  MpsTestParams params{1.5f, 2.5f, 99};

  auto key = kk::make(static_cast<Op>(2), Execution::Mps,
                      static_cast<kernel::Layout>(0), DType::F32,
                      static_cast<kernel::Variant>(0));

  EXPECT_TRUE(args.setParamsRaw(&params, sizeof(MpsTestParams), key));
  EXPECT_EQ(args.paramsSize(), sizeof(MpsTestParams));
  EXPECT_EQ(args.kernelKey(), key);

  // Verify we can retrieve the params
  MpsTestParams retrieved{};
  EXPECT_TRUE(args.getParams(retrieved, key));
  EXPECT_FLOAT_EQ(retrieved.alpha, 1.5f);
  EXPECT_FLOAT_EQ(retrieved.beta, 2.5f);
  EXPECT_EQ(retrieved.n, 99);
}

TEST(MpsKernelArgs, GetParamsFailsWithWrongType) {
  MpsArgs args;
  MpsTestParams params{1.0f, 2.0f, 100};

  auto key1 = kk::make(static_cast<Op>(1), Execution::Mps,
                       static_cast<kernel::Layout>(0), DType::F32,
                       static_cast<kernel::Variant>(0));

  auto key2 = kk::make(static_cast<Op>(2), // Different OpId
                       Execution::Mps, static_cast<kernel::Layout>(0),
                       DType::F32, static_cast<kernel::Variant>(0));

  args.setParams(params, key1);

  OtherMpsParams other{};
  EXPECT_FALSE(args.getParams(other, key2)); // Wrong key
}

TEST(MpsKernelArgs, HostFromCurrentContext) {
  // Setup: Configure MPS execution API
  namespace mps_api = ::orteaf::internal::execution::mps::api;
  namespace mps_context = ::orteaf::internal::execution_context::mps;
  namespace mps_platform = ::orteaf::internal::execution::mps::platform;

  auto *ops = new mps_platform::MpsSlowOpsImpl();
  const int device_count = ops->getDeviceCount();
  if (device_count <= 0) {
    delete ops;
    GTEST_SKIP() << "No MPS devices available";
  }

  mps_api::MpsExecutionApi::ExecutionManager::Config config{};
  config.slow_ops = ops;

  const auto capacity = static_cast<std::size_t>(device_count);
  auto &device_cfg = config.device_config;
  device_cfg.control_block_capacity = capacity;
  device_cfg.control_block_block_size = capacity;
  device_cfg.control_block_growth_chunk_size = 1;
  device_cfg.payload_capacity = capacity;
  device_cfg.payload_block_size = capacity;
  device_cfg.payload_growth_chunk_size = 1;

  auto configure_pool = [](auto &cfg) {
    cfg.control_block_capacity = 1;
    cfg.control_block_block_size = 1;
    cfg.control_block_growth_chunk_size = 1;
    cfg.payload_capacity = 1;
    cfg.payload_block_size = 1;
    cfg.payload_growth_chunk_size = 1;
  };
  configure_pool(device_cfg.command_queue_config);
  configure_pool(device_cfg.event_config);
  configure_pool(device_cfg.fence_config);
  configure_pool(device_cfg.heap_config);
  configure_pool(device_cfg.library_config);
  configure_pool(device_cfg.graph_config);

  try {
    mps_api::MpsExecutionApi::configure(config);
  } catch (const std::exception &ex) {
    delete ops;
    GTEST_SKIP() << "Failed to configure MPS: " << ex.what();
  }
  mps_context::reset();

  {
    // Test basic MPS kernel args usage
    MpsArgs args;
    EXPECT_EQ(args.storageCount(), 0);
  }

  // Teardown
  mps_context::reset();
  mps_api::MpsExecutionApi::shutdown();
}

// ============================================================
// Type-erased KernelArgs with MPS tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs;

TEST(KernelArgsMps, EraseFromMpsKernelArgs) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));
  EXPECT_TRUE(args.valid());
}

TEST(KernelArgsMps, TryAsMpsKernelArgs) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  auto *ptr = args.tryAs<MpsArgs>();
  EXPECT_NE(ptr, nullptr);
}

TEST(KernelArgsMps, ExecutionReturnsCorrectBackend) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  EXPECT_EQ(args.execution(), orteaf::internal::execution::Execution::Mps);
}

TEST(KernelArgsMps, VisitPattern) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  bool visited_mps = false;
  args.visit([&](auto &ka) {
    using T = std::decay_t<decltype(ka)>;
    if constexpr (std::is_same_v<T, MpsArgs>) {
      visited_mps = true;
    }
  });

  EXPECT_TRUE(visited_mps);
}

TEST(KernelArgsMps, TryAsWrongTypeReturnsNull) {
  using CpuArgs = kernel::cpu::CpuKernelArgs;

  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  // Trying to get as CPU should return nullptr
  auto *ptr = args.tryAs<CpuArgs>();
  EXPECT_EQ(ptr, nullptr);
}
