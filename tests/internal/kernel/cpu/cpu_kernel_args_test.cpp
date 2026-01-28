#include "orteaf/internal/kernel/core/access.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/param/param.h"
#include "orteaf/internal/kernel/param/param_id.h"
#include "orteaf/internal/kernel/param/param_key.h"
#include "orteaf/internal/kernel/storage/operand_id.h"
#include "orteaf/internal/kernel/storage/operand_key.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;

// ============================================================
// Test Fixture for KernelArgs (CPU context)
// ============================================================

class KernelArgsCpuContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Configure CPU execution API
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
    cpu_api::CpuExecutionApi::configure(config);
    ::orteaf::internal::execution_context::cpu::reset();
  }

  void TearDown() override {
    // Cleanup
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    ::orteaf::internal::execution_context::cpu::reset();
    cpu_api::CpuExecutionApi::shutdown();
  }
};

// ============================================================
// Access enum tests
// ============================================================

TEST(Access, EnumValues) {
  EXPECT_EQ(static_cast<uint8_t>(kernel::Access::None), 0);
  EXPECT_EQ(static_cast<uint8_t>(kernel::Access::Read), 1);
  EXPECT_EQ(static_cast<uint8_t>(kernel::Access::Write), 2);
  EXPECT_EQ(static_cast<uint8_t>(kernel::Access::ReadWrite), 3);
}

// ============================================================
// KernelArgs tests (CPU context available)
// ============================================================

using KernelArgsType = kernel::KernelArgs;

TEST_F(KernelArgsCpuContextTest, HostDefaultConstruct) {
  KernelArgsType host;
  EXPECT_EQ(host.storageCount(), 0);
  EXPECT_EQ(host.paramList().size(), 0);
}

TEST_F(KernelArgsCpuContextTest, AddAndFindParams) {
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

TEST_F(KernelArgsCpuContextTest, AddAndFindScopedParam) {
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

TEST_F(KernelArgsCpuContextTest, FindNonExistentParam) {
  KernelArgsType args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));

  const auto *param = args.findParam(kernel::ParamId::Beta);
  EXPECT_EQ(param, nullptr);
}

TEST_F(KernelArgsCpuContextTest, ClearParams) {
  KernelArgsType args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));

  EXPECT_EQ(args.paramList().size(), 2);

  args.clearParams();
  EXPECT_EQ(args.paramList().size(), 0);
}

TEST_F(KernelArgsCpuContextTest, StorageManagement) {
  KernelArgsType args;
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_GE(args.storageCapacity(), 0u);

  // Test clearing
  args.clearStorages();
  EXPECT_EQ(args.storageCount(), 0);
}

TEST_F(KernelArgsCpuContextTest, AddStorageBeyondInlineCapacity) {
  KernelArgsType args;
  const std::size_t count = 24;
  for (std::size_t i = 0; i < count; ++i) {
    KernelArgsType::StorageLease lease;
    args.addStorage(kernel::OperandId::InOut, std::move(lease));
  }
  EXPECT_EQ(args.storageCount(), count);
  EXPECT_GE(args.storageCapacity(), count);
}

TEST_F(KernelArgsCpuContextTest, AddStorageLease) {
  KernelArgsType args;

  // Add a storage lease with OperandId
  KernelArgsType::StorageLease lease;
  args.addStorage(kernel::OperandId::InOut, std::move(lease));

  EXPECT_EQ(args.storageCount(), 1);

  // Verify we can find the storage by ID
  const auto *binding = args.findStorage(kernel::OperandId::InOut);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding->key.id, kernel::OperandId::InOut);
  EXPECT_EQ(binding->key.role, kernel::Role::Data);
}

TEST_F(KernelArgsCpuContextTest, ParamListIteration) {
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

TEST_F(KernelArgsCpuContextTest, AddParamBeyondInlineCapacity) {
  KernelArgsType args;
  const std::size_t count = 24;
  for (std::size_t i = 0; i < count; ++i) {
    args.addParam(kernel::Param(kernel::ParamId::Alpha, static_cast<float>(i)));
  }
  EXPECT_EQ(args.paramList().size(), count);
  EXPECT_GE(args.paramList().capacity(), count);
}

TEST_F(KernelArgsCpuContextTest, HostFromCurrentContext) {
  // Build KernelArgs from the current CPU context
  auto ctx = kernel::ContextAny::erase(
      ::orteaf::internal::execution_context::cpu::currentContext());
  KernelArgsType args(std::move(ctx));
  EXPECT_TRUE(args.valid());
}

// ============================================================
// Type-erased KernelArgs tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs;

TEST(KernelArgs, DefaultConstructedIsInvalid) {
  TypeErasedArgs args;
  EXPECT_FALSE(args.valid());
}

TEST_F(KernelArgsCpuContextTest, ContextFromCpuContext) {
  auto ctx = kernel::ContextAny::erase(
      ::orteaf::internal::execution_context::cpu::Context{});
  TypeErasedArgs args(std::move(ctx));

  EXPECT_TRUE(args.valid());
  EXPECT_EQ(args.execution(), orteaf::internal::execution::Execution::Cpu);
  auto *cpu_ctx =
      args.context()
          .tryAs<::orteaf::internal::execution_context::cpu::Context>();
  EXPECT_NE(cpu_ctx, nullptr);
}

TEST(KernelArgs, ContextVisitOnInvalid) {
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
