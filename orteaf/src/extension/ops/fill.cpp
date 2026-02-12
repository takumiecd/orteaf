#include "orteaf/extension/ops/fill.h"

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/dispatch/dispatcher.h>
#include <orteaf/internal/kernel/param/param.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "op_common.h"

namespace orteaf::extension::ops {

namespace {

using Tensor = ::orteaf::user::tensor::Tensor;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;

}  // namespace

void fill(Tensor &output, double value) {
  detail::ensureValidTensor(output, "fill");
  const auto *impl = detail::requireDenseImpl(output, "fill");

  if (impl->execution() == Execution::Mps && impl->dtype() != ::orteaf::internal::DType::F32) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "fill: MPS currently supports only F32");
  }

  auto args = detail::makeArgsForCpuOrMps(impl->execution(), "fill");

  impl->bindAllArgs(args, kernel::OperandId::Output);
  args.addParam(kernel::Param(kernel::ParamId::Value, value));

  kernel::KeyRequest request{::orteaf::internal::ops::Op::Fill, impl->dtype(),
                             detail::architectureForArgs(args, "fill")};

  kernel::dispatch::Dispatcher dispatcher;
  auto result = dispatcher.dispatch(request, args);
  if (result.notFound()) {
    error::throwError(error::OrteafErrc::OperationFailed,
                      "Fill kernel not found for requested dtype");
  }
  if (result.failed()) {
    error::throwError(error::OrteafErrc::OperationFailed,
                      "Fill kernel execution failed");
  }
}

}  // namespace orteaf::extension::ops
