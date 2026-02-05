#include "orteaf/extension/ops/print.h"

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/core/context_any.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/dispatch/dispatcher.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

namespace orteaf::extension::ops {

namespace {

using Tensor = ::orteaf::user::tensor::Tensor;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;
using Architecture = ::orteaf::internal::architecture::Architecture;

void ensureValidTensor(const Tensor &tensor) {
  if (!tensor.valid()) {
    error::throwError(error::OrteafErrc::InvalidState, "Tensor is not valid");
  }
}

const DenseTensorImpl *requireDenseImpl(const Tensor &tensor) {
  const auto *lease = tensor.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease) || lease->operator->() == nullptr) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "Print op requires a dense tensor");
  }
  return lease->operator->();
}

kernel::KernelArgs makeArgsForExecution(Execution execution) {
  if (execution != Execution::Cpu) {
    error::throwError(error::OrteafErrc::ExecutionUnavailable,
                      "Print op currently supports CPU execution only");
  }
  auto cpu_context =
      ::orteaf::internal::execution_context::cpu::currentContext();
  return kernel::KernelArgs(kernel::ContextAny::erase(cpu_context));
}

}  // namespace

void print(const Tensor &input) {
  ensureValidTensor(input);
  const auto *impl = requireDenseImpl(input);

  if (!impl->isContiguous() || impl->offset() != 0) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "Print op currently requires contiguous tensors");
  }

  auto args = makeArgsForExecution(impl->execution());
  impl->bindAllArgs(args, kernel::OperandId::Input0);

  kernel::KeyRequest request{::orteaf::internal::ops::Op::Print, impl->dtype(),
                             Architecture::CpuGeneric};

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
