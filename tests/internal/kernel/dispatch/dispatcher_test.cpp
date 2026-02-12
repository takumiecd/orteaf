#include "orteaf/internal/kernel/dispatch/dispatcher.h"

#include <gtest/gtest.h>

#include "orteaf/internal/execution/cpu/api/cpu_execution_api.h"
#include "orteaf/internal/execution_context/cpu/context.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"
#include "orteaf/internal/kernel/api/kernel_registry_api.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/kernel_key.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"
#include "orteaf/internal/kernel/core/key_components.h"

namespace dispatch = orteaf::internal::kernel::dispatch;
namespace kernel = orteaf::internal::kernel;
namespace api = orteaf::internal::kernel::api;
using Architecture = orteaf::internal::architecture::Architecture;
using DType = orteaf::internal::DType;
using Op = orteaf::internal::ops::Op;

namespace {

// Test execution counter
static int g_test_execution_count = 0;

// Mock execute function for testing (now accepts KernelBase reference)
void mockExecuteFunc(
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase &,
    kernel::KernelArgs &args) {
  (void)args;
  ++g_test_execution_count;
}

// Helper to create a test metadata with CPU base
kernel::core::KernelMetadataLease makeTestMetadataWithCpuBase() {
  using CpuExecutionApi =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi;
  auto metadata_lease = CpuExecutionApi::acquireKernelMetadata(mockExecuteFunc);
  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

// Helper to create a test metadata
kernel::core::KernelMetadataLease makeTestMetadata() {
  return kernel::core::KernelMetadataLease{};
}

// Helper to create a test metadata with execute function
kernel::core::KernelMetadataLease makeTestMetadataWithExecute() {
  return makeTestMetadataWithCpuBase();
}

// Helper to register a test kernel (without execute)
void registerTestKernel(Op op, Architecture arch, kernel::Layout layout,
                        DType dtype, kernel::Variant variant) {
  auto key = kernel::kernel_key::make(op, arch, layout, dtype, variant);
  auto metadata = makeTestMetadata();
  api::KernelRegistryApi::registerKernel(key, std::move(metadata));
}

// Helper to register a test kernel with execute function
void registerTestKernelWithExecute(Op op, Architecture arch,
                                   kernel::Layout layout, DType dtype,
                                   kernel::Variant variant) {
  auto key = kernel::kernel_key::make(op, arch, layout, dtype, variant);
  auto metadata = makeTestMetadataWithExecute();
  api::KernelRegistryApi::registerKernel(key, std::move(metadata));
}

// Test fixture for Dispatcher tests
class DispatcherTest : public ::testing::Test {
protected:
  void SetUp() override {
    g_test_execution_count = 0;

    // Configure CPU execution API
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    cpu_api::CpuExecutionApi::ExecutionManager::Config config{};
    cpu_api::CpuExecutionApi::configure(config);
    ::orteaf::internal::execution_context::cpu::reset();

    // Clear the global registry before each test
    api::KernelRegistryApi::clear();
  }

  void TearDown() override {
    // Clean up the registry after each test
    api::KernelRegistryApi::clear();

    // Cleanup CPU execution API
    namespace cpu_api = ::orteaf::internal::execution::cpu::api;
    ::orteaf::internal::execution_context::cpu::reset();
    cpu_api::CpuExecutionApi::shutdown();
  }

  // Helper to create a basic KernelArgs with CPU context
  kernel::KernelArgs makeArgs() {
    auto cpu_context =
        orteaf::internal::execution_context::cpu::currentContext();
    auto ctx = kernel::ContextAny::erase(cpu_context);
    return kernel::KernelArgs(std::move(ctx));
  }
};

// ============================================================
// Basic construction tests
// ============================================================

TEST_F(DispatcherTest, DefaultConstruction) {
  dispatch::Dispatcher dispatcher;
  // Just verify it constructs without error
  SUCCEED();
}

// ============================================================
// DispatchResult tests
// ============================================================

TEST_F(DispatcherTest, DispatchResultSuccess) {
  dispatch::DispatchResult result{dispatch::DispatchStatus::Success};

  EXPECT_TRUE(result.success());
  EXPECT_FALSE(result.notFound());
  EXPECT_FALSE(result.failed());
}

TEST_F(DispatcherTest, DispatchResultNotFound) {
  dispatch::DispatchResult result{dispatch::DispatchStatus::NotFound};

  EXPECT_FALSE(result.success());
  EXPECT_TRUE(result.notFound());
  EXPECT_FALSE(result.failed());
}

TEST_F(DispatcherTest, DispatchResultExecutionError) {
  dispatch::DispatchResult result{dispatch::DispatchStatus::ExecutionError};

  EXPECT_FALSE(result.success());
  EXPECT_FALSE(result.notFound());
  EXPECT_TRUE(result.failed());
}

// ============================================================
// Resolve tests
// ============================================================

TEST_F(DispatcherTest, ResolveKernelNotFound) {
  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();
  auto *entry = dispatcher.resolve(request, args);

  EXPECT_EQ(entry, nullptr);
}

TEST_F(DispatcherTest, ResolveKernelFound) {
  // Register a test kernel
  registerTestKernel(static_cast<Op>(1), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F32,
                     static_cast<kernel::Variant>(0));

  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();
  auto *entry = dispatcher.resolve(request, args);

  EXPECT_NE(entry, nullptr);
}

// ============================================================
// Dispatch tests
// ============================================================

TEST_F(DispatcherTest, DispatchKernelNotFound) {
  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();
  auto result = dispatcher.dispatch(request, args);

  EXPECT_TRUE(result.notFound());
  EXPECT_FALSE(result.success());
  EXPECT_EQ(g_test_execution_count, 0);
}

TEST_F(DispatcherTest, DispatchKernelSuccess) {
  // Manually create and register a kernel with CPU base
  auto key =
      kernel::kernel_key::make(static_cast<Op>(1), Architecture::CpuGeneric,
                               static_cast<kernel::Layout>(0), DType::F32,
                               static_cast<kernel::Variant>(0));

  // Register via metadata and rebuild on lookup.
  registerTestKernelWithExecute(static_cast<Op>(1), Architecture::CpuGeneric,
                                static_cast<kernel::Layout>(0), DType::F32,
                                static_cast<kernel::Variant>(0));

  auto &registry = api::KernelRegistryApi::instance();
  auto *lookup_entry = registry.lookup(key);
  ASSERT_NE(lookup_entry, nullptr);

  // Now dispatch should succeed
  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();

  // Verify execution count starts at 0
  EXPECT_EQ(g_test_execution_count, 0);

  // Dispatch the kernel
  auto result = dispatcher.dispatch(request, args);

  // Verify successful dispatch
  EXPECT_TRUE(result.success());
  EXPECT_FALSE(result.notFound());
  EXPECT_FALSE(result.failed());

  // Verify the kernel was executed
  EXPECT_EQ(g_test_execution_count, 1);
}

TEST_F(DispatcherTest, DispatchKernelMultipleTimes) {
  // Register and set up kernel with CPU base
  registerTestKernelWithExecute(static_cast<Op>(1), Architecture::CpuGeneric,
                                static_cast<kernel::Layout>(0), DType::F32,
                                static_cast<kernel::Variant>(0));

  // Dispatch multiple times
  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();

  EXPECT_EQ(g_test_execution_count, 0);

  auto result1 = dispatcher.dispatch(request, args);
  EXPECT_TRUE(result1.success());
  EXPECT_EQ(g_test_execution_count, 1);

  auto result2 = dispatcher.dispatch(request, args);
  EXPECT_TRUE(result2.success());
  EXPECT_EQ(g_test_execution_count, 2);

  auto result3 = dispatcher.dispatch(request, args);
  EXPECT_TRUE(result3.success());
  EXPECT_EQ(g_test_execution_count, 3);
}

TEST_F(DispatcherTest, DispatchKernelWithoutValidBase) {
  // Register a kernel with execute function and rely on metadata rebuild
  registerTestKernelWithExecute(static_cast<Op>(1), Architecture::CpuGeneric,
                                static_cast<kernel::Layout>(0), DType::F32,
                                static_cast<kernel::Variant>(0));

  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();

  // Dispatch the kernel
  auto result = dispatcher.dispatch(request, args);

  // Metadata rebuild provides a valid base, so dispatch succeeds
  EXPECT_TRUE(result.success());
  EXPECT_FALSE(result.failed());
  EXPECT_FALSE(result.notFound());

  // Execute function should be called once
  EXPECT_EQ(g_test_execution_count, 1);
}

// ============================================================
// Integration with KernelRegistry API
// ============================================================

TEST_F(DispatcherTest, IntegrationWithRegistryAPI) {
  // Use the registry API directly
  auto key =
      kernel::kernel_key::make(static_cast<Op>(1), Architecture::CpuGeneric,
                               static_cast<kernel::Layout>(0), DType::F32,
                               static_cast<kernel::Variant>(0));

  // Register via API
  auto metadata = makeTestMetadata();
  api::KernelRegistryApi::registerKernel(key, std::move(metadata));

  // Verify it's in the registry
  EXPECT_TRUE(api::KernelRegistryApi::containsKernel(key));

  // Now try to resolve via dispatcher
  dispatch::Dispatcher dispatcher;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();
  auto *entry = dispatcher.resolve(request, args);

  EXPECT_NE(entry, nullptr);
}

// ============================================================
// Multiple kernel tests
// ============================================================

TEST_F(DispatcherTest, MultipleKernelsDifferentOps) {
  // Register kernels for different operations
  registerTestKernel(static_cast<Op>(1), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F32,
                     static_cast<kernel::Variant>(0));

  registerTestKernel(static_cast<Op>(2), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F32,
                     static_cast<kernel::Variant>(0));

  dispatch::Dispatcher dispatcher;
  auto args = makeArgs();

  // Resolve Op 1
  kernel::KeyRequest request1{static_cast<Op>(1), DType::F32,
                              Architecture::CpuGeneric};
  auto *entry1 = dispatcher.resolve(request1, args);
  EXPECT_NE(entry1, nullptr);

  // Resolve Op 2
  kernel::KeyRequest request2{static_cast<Op>(2), DType::F32,
                              Architecture::CpuGeneric};
  auto *entry2 = dispatcher.resolve(request2, args);
  EXPECT_NE(entry2, nullptr);

  // Resolve non-existent Op 3
  kernel::KeyRequest request3{static_cast<Op>(3), DType::F32,
                              Architecture::CpuGeneric};
  auto *entry3 = dispatcher.resolve(request3, args);
  EXPECT_EQ(entry3, nullptr);
}

TEST_F(DispatcherTest, MultipleKernelsDifferentDTypes) {
  // Register kernels for different data types
  registerTestKernel(static_cast<Op>(1), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F32,
                     static_cast<kernel::Variant>(0));

  registerTestKernel(static_cast<Op>(1), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F64,
                     static_cast<kernel::Variant>(0));

  dispatch::Dispatcher dispatcher;
  auto args = makeArgs();

  // Resolve F32
  kernel::KeyRequest request1{static_cast<Op>(1), DType::F32,
                              Architecture::CpuGeneric};
  auto *entry1 = dispatcher.resolve(request1, args);
  EXPECT_NE(entry1, nullptr);

  // Resolve F64
  kernel::KeyRequest request2{static_cast<Op>(1), DType::F64,
                              Architecture::CpuGeneric};
  auto *entry2 = dispatcher.resolve(request2, args);
  EXPECT_NE(entry2, nullptr);
}

// ============================================================
// Registry singleton tests
// ============================================================

TEST_F(DispatcherTest, RegistrySingletonConsistency) {
  // Register a kernel
  registerTestKernel(static_cast<Op>(1), Architecture::CpuGeneric,
                     static_cast<kernel::Layout>(0), DType::F32,
                     static_cast<kernel::Variant>(0));

  // Create two dispatchers
  dispatch::Dispatcher dispatcher1;
  dispatch::Dispatcher dispatcher2;

  kernel::KeyRequest request{static_cast<Op>(1), DType::F32,
                             Architecture::CpuGeneric};

  auto args = makeArgs();

  // Both should resolve to the same kernel (from singleton registry)
  auto *entry1 = dispatcher1.resolve(request, args);
  auto *entry2 = dispatcher2.resolve(request, args);

  EXPECT_NE(entry1, nullptr);
  EXPECT_NE(entry2, nullptr);
  EXPECT_EQ(entry1, entry2); // Same pointer from singleton
}

} // namespace
