#include "orteaf/internal/kernel/access.h"
#include "orteaf/internal/kernel/kernel_args.h"
#include "orteaf/internal/kernel/kernel_key.h"
#include "orteaf/internal/kernel/mps/mps_kernel_args.h"
#include "orteaf/internal/kernel/param.h"
#include "orteaf/internal/kernel/param_id.h"

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
// MpsKernelArgs tests
// ============================================================

using MpsArgs = kernel::mps::MpsKernelArgs;

TEST(MpsKernelArgs, HostDefaultConstruct) {
  MpsArgs host;
  EXPECT_EQ(host.storageCount(), 0);
  EXPECT_EQ(host.paramList().size(), 0);
}

TEST(MpsKernelArgs, AddAndFindParams) {
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

TEST(MpsKernelArgs, FindNonExistentParam) {
  MpsArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));

  const auto *param = args.findParam(kernel::ParamId::Beta);
  EXPECT_EQ(param, nullptr);
}

TEST(MpsKernelArgs, ClearParams) {
  MpsArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));

  EXPECT_EQ(args.paramList().size(), 2);

  args.clearParams();
  EXPECT_EQ(args.paramList().size(), 0);
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

TEST(MpsKernelArgs, ParamListIteration) {
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
    EXPECT_EQ(args.paramList().size(), 0);
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
