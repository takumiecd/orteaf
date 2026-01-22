#include "orteaf/internal/kernel/access.h"
#include "orteaf/internal/kernel/cpu/cpu_kernel_args.h"
#include "orteaf/internal/kernel/kernel_args.h"
#include "orteaf/internal/kernel/param.h"
#include "orteaf/internal/kernel/param_id.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;
using Execution = orteaf::internal::execution::Execution;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

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
// CpuKernelArgs tests
// ============================================================

using CpuArgs = kernel::cpu::CpuKernelArgs;

TEST(CpuKernelArgs, HostDefaultConstruct) {
  CpuArgs host;
  EXPECT_EQ(host.storageCount(), 0);
  EXPECT_EQ(host.paramList().size(), 0);
}

TEST(CpuKernelArgs, AddAndFindParams) {
  CpuArgs args;

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

TEST(CpuKernelArgs, FindNonExistentParam) {
  CpuArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));

  const auto *param = args.findParam(kernel::ParamId::Beta);
  EXPECT_EQ(param, nullptr);
}

TEST(CpuKernelArgs, ClearParams) {
  CpuArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));

  EXPECT_EQ(args.paramList().size(), 2);

  args.clearParams();
  EXPECT_EQ(args.paramList().size(), 0);
}

TEST(CpuKernelArgs, StorageManagement) {
  CpuArgs args;
  EXPECT_EQ(args.storageCount(), 0);
  EXPECT_EQ(args.storageCapacity(), CpuArgs::kMaxBindings);

  // Test clearing
  args.clearStorages();
  EXPECT_EQ(args.storageCount(), 0);
}

TEST(CpuKernelArgs, AddStorageLease) {
  CpuArgs args;

  // Add a storage lease (using default-constructed lease for now)
  kernel::cpu::CpuKernelArgs::StorageLease lease;
  args.addStorageLease(std::move(lease), kernel::Access::ReadWrite);

  EXPECT_EQ(args.storageCount(), 1);
  EXPECT_EQ(args.storageAccessAt(0), kernel::Access::ReadWrite);
}

TEST(CpuKernelArgs, ParamListIteration) {
  CpuArgs args;
  args.addParam(kernel::Param(kernel::ParamId::Alpha, 1.0f));
  args.addParam(kernel::Param(kernel::ParamId::Beta, 2.0f));
  args.addParam(kernel::Param(kernel::ParamId::Count, 42));

  int count = 0;
  for (const auto &param : args.paramList()) {
    ++count;
  }
  EXPECT_EQ(count, 3);
}

TEST(CpuKernelArgs, HostFromCurrentContext) {
  // Setup: Configure CPU execution API
  namespace cpu_api = ::orteaf::internal::execution::cpu::api;
  cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
  cpu_api::CpuExecutionApi::configure(config);
  ::orteaf::internal::execution_context::cpu::reset();

  {
    // fromCurrentContext should work with CPU args
    CpuArgs args;
    EXPECT_EQ(args.storageCount(), 0);
    EXPECT_EQ(args.paramList().size(), 0);
  }

  // Teardown
  ::orteaf::internal::execution_context::cpu::reset();
  cpu_api::CpuExecutionApi::shutdown();
}

// ============================================================
// Type-erased KernelArgs tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs;

TEST(KernelArgs, DefaultConstructedIsInvalid) {
  TypeErasedArgs args;
  EXPECT_FALSE(args.valid());
}

TEST(KernelArgs, EraseFromCpuKernelArgs) {
  CpuArgs cpu_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(cpu_args));
  EXPECT_TRUE(args.valid());
}

TEST(KernelArgs, TryAsCpuKernelArgs) {
  CpuArgs cpu_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(cpu_args));

  auto *ptr = args.tryAs<CpuArgs>();
  EXPECT_NE(ptr, nullptr);
}

TEST(KernelArgs, ExecutionReturnsCorrectBackend) {
  CpuArgs cpu_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(cpu_args));

  EXPECT_EQ(args.execution(), orteaf::internal::execution::Execution::Cpu);
}

TEST(KernelArgs, VisitPattern) {
  CpuArgs cpu_args;
  TypeErasedArgs args = TypeErasedArgs::erase(std::move(cpu_args));

  bool visited_cpu = false;
  args.visit([&](auto &ka) {
    using T = std::decay_t<decltype(ka)>;
    if constexpr (std::is_same_v<T, CpuArgs>) {
      visited_cpu = true;
    }
  });

  EXPECT_TRUE(visited_cpu);
}

TEST(KernelArgs, VisitPatternOnInvalid) {
  TypeErasedArgs args;

  bool visited_monostate = false;
  args.visit([&](auto &ka) {
    using T = std::decay_t<decltype(ka)>;
    if constexpr (std::is_same_v<T, std::monostate>) {
      visited_monostate = true;
    }
  });

  EXPECT_TRUE(visited_monostate);
}
