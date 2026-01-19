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
  float alpha{0.0f};
  float beta{0.0f};
  int n{0};
};
static_assert(std::is_trivially_copyable_v<TestParams>);

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

using CpuArgs = kernel::cpu::CpuKernelArgs<4, TestParams>;

TEST(CpuKernelArgs, DeviceIsPOD) {
  static_assert(std::is_trivially_copyable_v<CpuArgs::Device>);
  static_assert(std::is_trivially_copyable_v<CpuArgs::Device::Binding>);
}

TEST(CpuKernelArgs, HostDefaultConstruct) {
  CpuArgs::Host host;
  EXPECT_EQ(host.storageCount(), 0);
}

TEST(CpuKernelArgs, DeviceDefaultConstruct) {
  CpuArgs::Device device{};
  EXPECT_EQ(device.binding_count, 0);
  EXPECT_FLOAT_EQ(device.params.alpha, 0.0f);
}

TEST(CpuKernelArgs, ToDeviceBasic) {
  CpuArgs::Host host;
  TestParams params{1.0f, 2.0f, 100};

  auto device = CpuArgs::toDevice(host, params);
  EXPECT_EQ(device.binding_count, 0);
  EXPECT_FLOAT_EQ(device.params.alpha, 1.0f);
  EXPECT_FLOAT_EQ(device.params.beta, 2.0f);
  EXPECT_EQ(device.params.n, 100);
}

TEST(CpuKernelArgs, HostFromCurrentContext) {
  // Setup: Configure CPU execution API
  namespace cpu_api = ::orteaf::internal::execution::cpu::api;
  cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
  cpu_api::CpuExecutionApi::configure(config);
  ::orteaf::internal::execution_context::cpu::reset();

  {
    // fromCurrentContext should return a Host with the current thread-local
    // context
    auto host = CpuArgs::Host::fromCurrentContext();
    // Should be able to use the host normally
    EXPECT_EQ(host.storageCount(), 0);
  } // host destroyed here, releasing context references

  // Teardown
  ::orteaf::internal::execution_context::cpu::reset();
  cpu_api::CpuExecutionApi::shutdown();
}

// ============================================================
// Type-erased KernelArgs tests
// ============================================================

using TypeErasedArgs = kernel::KernelArgs<4, TestParams>;

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
