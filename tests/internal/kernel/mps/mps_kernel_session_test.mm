#include <gtest/gtest.h>

#include <exception>
#include <string>
#include <system_error>
#include <utility>

#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#include <orteaf/internal/execution_context/mps/current_context.h>
#include <orteaf/internal/kernel/core/context_any.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session.h>

namespace kernel = ::orteaf::internal::kernel;
namespace mps_api = ::orteaf::internal::execution::mps::api;
namespace mps_context = ::orteaf::internal::execution_context::mps;
namespace mps_kernel = ::orteaf::internal::kernel::mps;

namespace {

struct MpsExecutionGuard {
  bool configured{false};
  std::string reason{};

  MpsExecutionGuard() {
    try {
      mps_api::MpsExecutionApi::configure();
      configured = true;
    } catch (const std::exception &ex) {
      reason = ex.what();
    }
  }

  ~MpsExecutionGuard() {
    mps_context::reset();
    mps_api::MpsExecutionApi::shutdown();
  }
};

TEST(MpsKernelSessionTest, BeginReturnsNulloptWithInvalidContext) {
  mps_api::MpsExecutionApi::shutdown();
  kernel::KernelArgs args;
  ::orteaf::internal::execution::mps::resource::MpsKernelBase base;

  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST(MpsKernelSessionTest, BeginReturnsNulloptWithContextWithoutQueue) {
  mps_api::MpsExecutionApi::shutdown();
  auto ctx_any = kernel::ContextAny::erase(mps_context::Context{});
  kernel::KernelArgs args(std::move(ctx_any));
  ::orteaf::internal::execution::mps::resource::MpsKernelBase base;

  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST(MpsKernelSessionTest, BeginReturnsNulloptWhenPipelineIsUnavailable) {
  MpsExecutionGuard guard;
  if (!guard.configured) {
    GTEST_SKIP() << "Failed to configure MPS execution: " << guard.reason;
  }

  kernel::KernelArgs args(
      kernel::ContextAny::erase(mps_context::currentContext()));
  ::orteaf::internal::execution::mps::resource::MpsKernelBase base;

  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  EXPECT_FALSE(session.has_value());
}

TEST(MpsKernelSessionTest, BeginSucceedsWithConfiguredPipeline) {
  MpsExecutionGuard guard;
  if (!guard.configured) {
    GTEST_SKIP() << "Failed to configure MPS execution: " << guard.reason;
  }

  kernel::KernelArgs args(
      kernel::ContextAny::erase(mps_context::currentContext()));
  auto *context = args.context().tryAs<mps_context::Context>();
  ASSERT_NE(context, nullptr);

  mps_api::MpsExecutionApi::KernelKeys keys;
  keys.pushBack(mps_api::MpsExecutionApi::KernelKey{
      mps_api::MpsExecutionApi::LibraryKey::Named("fill_kernel"),
      mps_api::MpsExecutionApi::FunctionKey::Named("orteaf_fill_strided_f32")});

  auto lease =
      mps_api::MpsExecutionApi::executionManager().kernelBaseManager().acquire(
          keys);
  ASSERT_TRUE(lease);
  auto *base = lease.operator->();
  ASSERT_NE(base, nullptr);
  if (!lease->ensurePipelines(context->device)) {
    GTEST_SKIP() << "Failed to ensure MPS pipelines for test session";
  }

  auto session = mps_kernel::MpsKernelSession::begin(*base, args, 0);
  ASSERT_TRUE(session.has_value());
  EXPECT_NE(session->encoder(), nullptr);
  EXPECT_NE(session->commandBuffer(), nullptr);
}

TEST(MpsKernelSessionTest, DestructorCommitsCommandBuffer) {
  MpsExecutionGuard guard;
  if (!guard.configured) {
    GTEST_SKIP() << "Failed to configure MPS execution: " << guard.reason;
  }

  kernel::KernelArgs args(
      kernel::ContextAny::erase(mps_context::currentContext()));
  auto *context = args.context().tryAs<mps_context::Context>();
  ASSERT_NE(context, nullptr);

  mps_api::MpsExecutionApi::KernelKeys keys;
  keys.pushBack(mps_api::MpsExecutionApi::KernelKey{
      mps_api::MpsExecutionApi::LibraryKey::Named("fill_kernel"),
      mps_api::MpsExecutionApi::FunctionKey::Named("orteaf_fill_strided_f32")});

  auto lease =
      mps_api::MpsExecutionApi::executionManager().kernelBaseManager().acquire(
          keys);
  ASSERT_TRUE(lease);
  auto *base = lease.operator->();
  ASSERT_NE(base, nullptr);
  if (!lease->ensurePipelines(context->device)) {
    GTEST_SKIP() << "Failed to ensure MPS pipelines for commit test";
  }

  mps_kernel::MpsKernelSession::MpsCommandBuffer_t command_buffer = nullptr;
  {
    auto session = mps_kernel::MpsKernelSession::begin(*base, args, 0);
    ASSERT_TRUE(session.has_value());
    command_buffer = session->commandBuffer();
    ASSERT_NE(command_buffer, nullptr);
  }

  mps_kernel::MpsKernelSession::Ops::waitUntilCompleted(command_buffer);
  SUCCEED();
}

} // namespace
