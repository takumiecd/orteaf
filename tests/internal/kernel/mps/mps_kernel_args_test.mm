#include "orteaf/internal/kernel/access.h"
#include "orteaf/internal/kernel/kernel_args.h"
#include "orteaf/internal/kernel/kernel_key.h"
#include "orteaf/internal/kernel/mps/mps_kernel_args.h"
#include "orteaf/internal/kernel/param.h"
#include "orteaf/internal/kernel/param_id.h"
#include "orteaf/internal/kernel/storage_id.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution_context/mps/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

// ============================================================
// Test Fixture for MPS Kernel Args
// ============================================================

class MpsKernelArgsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Configure MPS execution API
    namespace mps_api = ::orteaf::internal::execution::mps::api;
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
    ::orteaf::internal::execution_context::mps::reset();
  }

  void TearDown() override {
    // Cleanup
    namespace mps_api = ::orteaf::internal::execution::mps::api;
    ::orteaf::internal::execution_context::mps::reset();
    mps_api::MpsExecutionApi::shutdown();
  }
};

// ============================================================
// MpsKernelArgs tests
// ============================================================

using MpsArgs = kernel::mps::MpsKernelArgs;

TEST_F(MpsKernelArgsTest, HostDefaultConstruct) {
  MpsArgs host;
  EXPECT_EQ(host.storageCount(), 0);
  EXPECT_EQ(host.paramList().size(), 0);
}

TEST_F(MpsKernelArgsTest, AddAndFindParams) {
  MpsArgs args;

  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.5f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.5f));
  args.addParam(kernel::Param(kernel::ParamId::Count, 100));

  EXPECT_EQ(args.paramList().size(), 3);

  const auto *alpha_param = args.findParam(kernel::ParamId::Alpha);
  ASSERT_NE(alpha_param, nullptr);
  EXPECT_FLOAT_EQ(*alpha_param->tryGet<float>(), 1.5f);

  const auto *beta_param = args.findParam(kernel::ParamId::Beta);
  ASSERT_NE(beta_param, nullptr);
  EXPECT_FLOAT_EQ(*beta_param->tryGet<float>(), 2.5f);

  const auto *count_param = args.findParam(kernel::ParamId::Count);
  ASSERT_NE(count_param, nullptr);
  EXPECT_EQ(*count_param->tryGet<int>(), 100);
}

TEST_F(MpsKernelArgsTest, FindNonExistentParam) {
  MpsArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));

  const auto *param = args.findParam(kernel::ParamId::Beta);
  EXPECT_EQ(param, nullptr);
}

TEST_F(MpsKernelArgsTest, ClearParams) {
  MpsArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));

  EXPECT_EQ(args.paramList().size(), 2);

  args.clearParams();
  EXPECT_EQ(args.paramList().size(), 0);
}

TEST_F(MpsKernelArgsTest, StorageManagement) {
  MpsArgs args;
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_GE(args.storageCapacity(), MpsArgs::kMaxBindings);

  // Test clearing
  args.clearStorages();
  EXPECT_EQ(args.storageCount(), 0);
}

TEST_F(MpsKernelArgsTest, AddStorageBeyondInlineCapacity) {
  MpsArgs args;
  const std::size_t count = MpsArgs::kMaxBindings + 4;
  for (std::size_t i = 0; i < count; ++i) {
    MpsArgs::StorageLease lease;
    args.addStorage(kernel::StorageId::Input0, std::move(lease));
  }
  EXPECT_EQ(args.storageCount(), count);
  EXPECT_GE(args.storageCapacity(), count);
}

TEST_F(MpsKernelArgsTest, AddStorageLease) {
  MpsArgs args;

  // Add a storage lease with StorageId
  kernel::mps::MpsKernelArgs::StorageLease lease;
  args.addStorage(kernel::StorageId::Input0, std::move(lease));

  EXPECT_EQ(args.storageCount(), 1);

  // Verify we can find the storage by ID
  const auto *binding = args.findStorage(kernel::StorageId::Input0);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding->id, kernel::StorageId::Input0);
}

TEST_F(MpsKernelArgsTest, ParamListIteration) {
  MpsArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));
  args.addParam(kernel::Param(kernel::ParamId::Count, 42));

  int count = 0;
  for (const auto &param : args.paramList()) {
    ++count;
  }
  EXPECT_EQ(count, 3);
}

TEST_F(MpsKernelArgsTest, AddParamBeyondInlineCapacity) {
  MpsArgs args;
  const std::size_t count = MpsArgs::kMaxParams + 4;
  for (std::size_t i = 0; i < count; ++i) {
    args.addParam(kernel::Param(kernel::ParamId::Alpha,
                                static_cast<float>(i)));
  }
  EXPECT_EQ(args.paramList().size(), count);
  EXPECT_GE(args.paramList().capacity(), count);
}

TEST_F(MpsKernelArgsTest, HostFromCurrentContext) {
  // Test basic MPS kernel args usage with fromCurrentContext
  MpsArgs args = MpsArgs::fromCurrentContext();
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_EQ(args.paramList().size(), 0);
}

// ============================================================
// Type-erased KernelArgs with MPS tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs;

TEST_F(MpsKernelArgsTest, EraseFromMpsKernelArgs) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));
  EXPECT_TRUE(args.valid());
}

TEST_F(MpsKernelArgsTest, TryAsMpsKernelArgs) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  auto *ptr = args.tryAs<MpsArgs>();
  EXPECT_NE(ptr, nullptr);
}

TEST_F(MpsKernelArgsTest, ExecutionReturnsCorrectBackend) {
  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  EXPECT_EQ(args.execution(), orteaf::internal::execution::Execution::Mps);
}

TEST_F(MpsKernelArgsTest, VisitPattern) {
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

TEST_F(MpsKernelArgsTest, TryAsWrongTypeReturnsNull) {
  using CpuArgs = kernel::cpu::CpuKernelArgs;

  MpsArgs mps_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(mps_args));

  // Trying to get as CPU should return nullptr
  auto *ptr = args.tryAs<CpuArgs>();
  EXPECT_EQ(ptr, nullptr);
}
