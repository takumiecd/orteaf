#pragma once

#include <memory>
#include <type_traits>

#include <gmock/gmock.h>

#include <orteaf/internal/execution/mps/platform/mps_slow_ops.h>
<<<<<<<< HEAD:tests/internal/execution/mps/manager/testing/execution_ops_provider.h
#include <tests/internal/execution/mps/manager/testing/execution_mock.h>
========
#include <tests/internal/execution/mps/manager/testing/backend_mock.h>
>>>>>>>> 63ca5d9 (refactor: migrate internal runtime MPS and CUDA platform components to execution directory.):tests/internal/execution/mps/manager/testing/backend_ops_provider.h

namespace orteaf::tests::execution::mps::testing {

template <class ExecutionOpsT, bool IsMockV> struct ExecutionOpsProvider {
  using SlowOps = ExecutionOpsT;
  static constexpr bool is_mock = IsMockV;

  struct Context {
    Context() = default;
  };

  static void setUp(Context &) {}
  static void tearDown(Context &) {}
};

// Real provider uses MpsSlowOpsImpl
<<<<<<<< HEAD:tests/internal/execution/mps/manager/testing/execution_ops_provider.h
struct RealExecutionOpsProvider {
========
struct RealBackendOpsProvider {
>>>>>>>> 63ca5d9 (refactor: migrate internal runtime MPS and CUDA platform components to execution directory.):tests/internal/execution/mps/manager/testing/backend_ops_provider.h
  using SlowOps = ::orteaf::internal::execution::mps::platform::MpsSlowOpsImpl;
  static constexpr bool is_mock = false;

  struct Context {
    SlowOps ops;
  };

  static void setUp(Context &) {}
  static void tearDown(Context &) {}

  static SlowOps *getOps(Context &ctx) { return &ctx.ops; }
};

<<<<<<<< HEAD:tests/internal/execution/mps/manager/testing/execution_ops_provider.h
// Mock provider uses MpsExecutionOpsMock
struct MockExecutionOpsProvider {
  using SlowOps = ::orteaf::tests::execution::mps::MpsExecutionOpsMock;
========
// Mock provider uses MpsBackendOpsMock
struct MockBackendOpsProvider {
  using SlowOps = ::orteaf::tests::execution::mps::MpsBackendOpsMock;
>>>>>>>> 63ca5d9 (refactor: migrate internal runtime MPS and CUDA platform components to execution directory.):tests/internal/execution/mps/manager/testing/backend_ops_provider.h
  static constexpr bool is_mock = true;

  struct Context {
    ::testing::NiceMock<SlowOps> mock;
  };

  static void setUp(Context &) {}
  static void tearDown(Context &) {}

  static SlowOps *getOps(Context &ctx) { return &ctx.mock; }
  static SlowOps &mock(Context &ctx) { return ctx.mock; }
};

} // namespace orteaf::tests::execution::mps::testing
