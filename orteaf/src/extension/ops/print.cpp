#include "orteaf/extension/ops/print.h"

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/dispatch/dispatcher.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "op_common.h"

namespace orteaf::extension::ops {

namespace {

using Tensor = ::orteaf::user::tensor::Tensor;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;

}  // namespace

void print(const Tensor &input) {
  detail::ensureValidTensor(input, "print");
  const auto *impl = detail::requireDenseImpl(input, "print");

  if (!impl->isContiguous() || impl->offset() != 0) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "Print op currently requires contiguous tensors");
  }

  auto args = detail::makeArgsForCpuOnly(impl->execution(), "print");
  impl->bindAllArgs(args, kernel::OperandId::Input0);

  kernel::KeyRequest request{::orteaf::internal::ops::Op::Print, impl->dtype(),
                             detail::architectureForArgs(args, "print")};

  kernel::dispatch::Dispatcher dispatcher;
  auto result = dispatcher.dispatch(request, args);
  if (result.notFound()) {
    error::throwError(error::OrteafErrc::OperationFailed,
                      "Print kernel not found for requested dtype");
  }
  if (result.failed()) {
    error::throwError(error::OrteafErrc::OperationFailed,
                      "Print kernel execution failed");
  }
}

}  // namespace orteaf::extension::ops
