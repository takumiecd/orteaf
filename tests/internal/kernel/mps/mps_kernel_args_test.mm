#include "orteaf/internal/kernel/core/access.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/core/kernel_key.h"
#include "orteaf/internal/kernel/param/param.h"
#include "orteaf/internal/kernel/param/param_id.h"
#include "orteaf/internal/kernel/param/param_key.h"
#include "orteaf/internal/kernel/storage/operand_id.h"
#include "orteaf/internal/kernel/storage/operand_key.h"

#include "orteaf/internal/execution/mps/api/mps_execution_api.h"
#include "orteaf/internal/execution/mps/platform/mps_slow_ops.h"
#include "orteaf/internal/execution_context/mps/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;

// ============================================================
// Test Fixture for KernelArgs (MPS context)
// ============================================================

class KernelArgsMpsContextTest : public ::testing::Test {
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
// KernelArgs tests (MPS context available)
// ============================================================

using KernelArgsType = kernel::KernelArgs;

TEST_F(KernelArgsMpsContextTest, HostDefaultConstruct) {
  KernelArgsType host;
  EXPECT_EQ(host.storageCount(), 0);
  EXPECT_EQ(host.paramList().size(), 0);
}

TEST_F(KernelArgsMpsContextTest, AddAndFindParams) {
  KernelArgsType args;

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

TEST_F(KernelArgsMpsContextTest, AddAndFindScopedParam) {
  KernelArgsType args;

  const auto key = kernel::ParamKey::scoped(
      kernel::ParamId::Alpha,
      kernel::makeOperandKey(kernel::OperandId::Input0));
  args.addParam(kernel::Param(key, 3.5f));

  // Global lookup should not match scoped params.
  EXPECT_EQ(args.findParam(kernel::ParamId::Alpha), nullptr);

  const auto *param = args.findParam(key);
  ASSERT_NE(param, nullptr);
  EXPECT_FLOAT_EQ(*param->tryGet<float>(), 3.5f);
}

TEST_F(KernelArgsMpsContextTest, FindNonExistentParam) {
  KernelArgsType args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));

  const auto *param = args.findParam(kernel::ParamId::Beta);
  EXPECT_EQ(param, nullptr);
}

TEST_F(KernelArgsMpsContextTest, ClearParams) {
  KernelArgsType args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));

  EXPECT_EQ(args.paramList().size(), 2);

  args.clearParams();
  EXPECT_EQ(args.paramList().size(), 0);
}

TEST_F(KernelArgsMpsContextTest, StorageManagement) {
  KernelArgsType args;
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_GE(args.storageCapacity(), 0u);

  // Test clearing
  args.clearStorages();
  EXPECT_EQ(args.storageCount(), 0);
}

TEST_F(KernelArgsMpsContextTest, AddStorageBeyondInlineCapacity) {
  KernelArgsType args;
  const std::size_t count = 24;
  for (std::size_t i = 0; i < count; ++i) {
    KernelArgsType::StorageLease lease;
    args.addStorage(kernel::OperandId::Input0, std::move(lease));
  }
  EXPECT_EQ(args.storageCount(), count);
  EXPECT_GE(args.storageCapacity(), count);
}

TEST_F(KernelArgsMpsContextTest, AddStorageLease) {
  KernelArgsType args;

  // Add a storage lease with OperandId
  KernelArgsType::StorageLease lease;
  args.addStorage(kernel::OperandId::Input0, std::move(lease));

  EXPECT_EQ(args.storageCount(), 1);

  // Verify we can find the storage by ID
  const auto *binding = args.findStorage(kernel::OperandId::Input0);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding->key.id, kernel::OperandId::Input0);
  EXPECT_EQ(binding->key.role, kernel::Role::Data);
}

TEST_F(KernelArgsMpsContextTest, ParamListIteration) {
  KernelArgsType args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));
  args.addParam(kernel::Param(kernel::ParamId::Count, 42));

  int count = 0;
  for (const auto &param : args.paramList()) {
    ++count;
  }
  EXPECT_EQ(count, 3);
}

TEST_F(KernelArgsMpsContextTest, AddParamBeyondInlineCapacity) {
  KernelArgsType args;
  const std::size_t count = 24;
  for (std::size_t i = 0; i < count; ++i) {
    args.addParam(kernel::Param(kernel::ParamId::Alpha, static_cast<float>(i)));
  }
  EXPECT_EQ(args.paramList().size(), count);
  EXPECT_GE(args.paramList().capacity(), count);
}

TEST_F(KernelArgsMpsContextTest, HostFromCurrentContext) {
  // Build KernelArgs from the current MPS context
  auto ctx = kernel::ContextAny::erase(
      ::orteaf::internal::execution_context::mps::currentContext());
  KernelArgsType args(std::move(ctx));
  EXPECT_TRUE(args.valid());
}

// ============================================================
// Type-erased KernelArgs with MPS tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs;

TEST_F(KernelArgsMpsContextTest, ContextFromMpsContext) {
  auto ctx = kernel::ContextAny::erase(
      ::orteaf::internal::execution_context::mps::Context{});
  TypeErasedArgs args(std::move(ctx));

  EXPECT_TRUE(args.valid());
  EXPECT_EQ(args.execution(), orteaf::internal::execution::Execution::Mps);
  auto *mps_ctx =
      args.context()
          .tryAs<::orteaf::internal::execution_context::mps::Context>();
  EXPECT_NE(mps_ctx, nullptr);
}

TEST_F(KernelArgsMpsContextTest, ContextVisitOnInvalid) {
  TypeErasedArgs args;

  bool visited_monostate = false;
  args.context().visit([&](const auto &ctx) {
    using T = std::decay_t<decltype(ctx)>;
    if constexpr (std::is_same_v<T, std::monostate>) {
      visited_monostate = true;
    }
  });

  EXPECT_TRUE(visited_monostate);
}
