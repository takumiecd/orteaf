#include "orteaf/internal/kernel/access.h"
#include "orteaf/internal/kernel/cpu/cpu_kernel_args.h"
#include "orteaf/internal/kernel/kernel_args.h"

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace kernel = orteaf::internal::kernel;

// ============================================================
// Test Params struct (POD)
// ============================================================

struct TestParams {
  static constexpr std::uint64_t kTypeId = 1;
  float alpha{0.0f};
  float beta{0.0f};
  int n{0};
};
static_assert(std::is_trivially_copyable_v<TestParams>);

// OtherParams for type mismatch testing
struct OtherParams {
  static constexpr std::uint64_t kTypeId = 999;
  int value{0};
};
static_assert(std::is_trivially_copyable_v<OtherParams>);

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
}

TEST(CpuKernelArgs, SetAndGetParams) {
  CpuArgs args;
  TestParams params{1.0f, 2.0f, 100};

  EXPECT_TRUE(args.setParams(params));
  EXPECT_EQ(args.paramsSize(), sizeof(TestParams));
  EXPECT_EQ(args.paramsTypeId(), TestParams::kTypeId);

  TestParams retrieved{};
  EXPECT_TRUE(args.getParams(retrieved));
  EXPECT_FLOAT_EQ(retrieved.alpha, 1.0f);
  EXPECT_FLOAT_EQ(retrieved.beta, 2.0f);
  EXPECT_EQ(retrieved.n, 100);
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

TEST(CpuKernelArgs, ParamsRawAccessors) {
  CpuArgs args;
  TestParams params{3.14f, 2.71f, 42};

  EXPECT_TRUE(args.setParams(params));

  // Test raw data access
  const std::byte *data = args.paramsData();
  EXPECT_NE(data, nullptr);
  EXPECT_EQ(args.paramsSize(), sizeof(TestParams));
  EXPECT_EQ(args.paramsCapacity(), CpuArgs::kParamBytes);

  // Test non-const data access
  std::byte *mutable_data = args.paramsData();
  EXPECT_NE(mutable_data, nullptr);
}

TEST(CpuKernelArgs, SetParamsRaw) {
  CpuArgs args;
  TestParams params{1.5f, 2.5f, 99};

  EXPECT_TRUE(
      args.setParamsRaw(&params, sizeof(TestParams), TestParams::kTypeId));
  EXPECT_EQ(args.paramsSize(), sizeof(TestParams));
  EXPECT_EQ(args.paramsTypeId(), TestParams::kTypeId);

  // Verify we can retrieve the params
  TestParams retrieved{};
  EXPECT_TRUE(args.getParams(retrieved));
  EXPECT_FLOAT_EQ(retrieved.alpha, 1.5f);
  EXPECT_FLOAT_EQ(retrieved.beta, 2.5f);
  EXPECT_EQ(retrieved.n, 99);
}

TEST(CpuKernelArgs, GetParamsFailsWithWrongType) {
  CpuArgs args;
  TestParams params{1.0f, 2.0f, 100};
  args.setParams(params);

  OtherParams other{};
  EXPECT_FALSE(args.getParams(other));
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
