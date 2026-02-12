#include <orteaf/extension/ops/dense/dense_tensor_ops.h>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "detail/dense_op_common.h"

namespace orteaf::extension::ops::dense {

namespace {

namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;

} // namespace

void DenseTensorOps::fill(Tensor &output, double value) {
  constexpr const char *kOpName = "fill";
  detail::ensureValidTensor(output, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);

  if (output_impl->execution() == Execution::Mps &&
      output_impl->dtype() != ::orteaf::internal::DType::F32) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "fill: MPS currently supports only F32");
  }

  auto args = detail::makeArgsForCpuOrMps(output_impl->execution(), kOpName);
  output_impl->bindAllArgs(args, kernel::OperandId::Output);
  args.addParam(kernel::Param(kernel::ParamId::Value, value));

  const kernel::KeyRequest request{::orteaf::internal::ops::Op::Fill,
                                   output_impl->dtype(),
                                   detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args,
                          "Fill kernel not found for requested dtype",
                          "Fill kernel execution failed");
}

} // namespace orteaf::extension::ops::dense
