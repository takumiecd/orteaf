#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/param/param_key.h>
#include <orteaf/internal/kernel/storage/operand_key.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "tests/internal/kernel/mps/ops/fixtures/scoped_param_kernel.h"

namespace kernel = orteaf::internal::kernel;
namespace mps_kernel = orteaf::internal::kernel::mps;
namespace scoped_kernel = orteaf::extension::kernel::mps::ops;

namespace {

TEST(ScopedParamKernelTest, ParamSchemaIsScopedToInput0) {
  scoped_kernel::ScopedParamParams params;

  EXPECT_EQ(params.num_elements.kId, kernel::ParamId::NumElements);
  EXPECT_EQ(params.num_elements.kOperandKey.id, kernel::OperandId::Input0);
  EXPECT_EQ(params.num_elements.kOperandKey.role, kernel::Role::Data);
}

TEST(ScopedParamKernelTest, ExecuteExtractsScopedParam) {
  mps_kernel::MpsKernelBase base;
  kernel::KernelArgs args;

  const auto key = kernel::ParamKey::scoped(
      kernel::ParamId::NumElements,
      kernel::makeOperandKey(kernel::OperandId::Input0));
  args.addParam(kernel::Param(key, static_cast<std::uint32_t>(7)));

  auto entry = scoped_kernel::createScopedParamKernel();
  entry.execute(base, args);

  // Global lookup should not match scoped params.
  EXPECT_EQ(args.findParam(kernel::ParamId::NumElements), nullptr);

  const auto *count_param = args.findParam(kernel::ParamId::Count);
  ASSERT_NE(count_param, nullptr);
  EXPECT_EQ(*count_param->tryGet<std::size_t>(), 7u);
}

} // namespace
