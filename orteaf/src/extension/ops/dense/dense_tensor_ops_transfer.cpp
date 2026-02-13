#include <orteaf/extension/ops/dense/dense_tensor_ops.h>

#include <string>

#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "detail/dense_op_common.h"
#include "detail/dense_transfer_common.h"

namespace orteaf::extension::ops::dense {

namespace {

using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;
using Op = ::orteaf::internal::ops::Op;

void dispatchTransferKernel(const char *op_name, Op op, Execution execution,
                            const DenseTensorImpl *input_impl,
                            const DenseTensorImpl *output_impl,
                            const char *not_found_message,
                            const char *failed_message) {
  auto args = detail::makeArgsForCpuOrMpsOrCuda(execution, op_name);
  input_impl->bindAllArgs(args, kernel::OperandId::Input0);
  output_impl->bindAllArgs(args, kernel::OperandId::Output);

  const kernel::KeyRequest request{op, output_impl->dtype(),
                                   detail::architectureForArgs(args, op_name)};
  detail::dispatchOrThrow(request, args, not_found_message, failed_message);
}

} // namespace

void DenseTensorOps::copyHostToDevice(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyHostToDevice";
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(input_impl, Execution::Cpu, kOpName,
                                     "input");

  const auto output_exec = output_impl->execution();
  if (output_exec != Execution::Mps && output_exec != Execution::Cuda) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToDevice: output must be a device tensor");
  }
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 kOpName);
  detail::transfer::ensureRankSupported(input_impl, kOpName);
  detail::transfer::ensureRankSupported(output_impl, kOpName);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, kOpName, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, kOpName, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyHostToDevice: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  dispatchTransferKernel(kOpName, Op::CopyHostToDevice, output_exec, input_impl,
                         output_impl, "copyHostToDevice kernel not found",
                         "copyHostToDevice kernel execution failed");
}

void DenseTensorOps::copyDeviceToHost(Tensor &output, const Tensor &input) {
  constexpr const char *kOpName = "copyDeviceToHost";
  detail::ensureValidTensor(output, kOpName);
  detail::ensureValidTensor(input, kOpName);
  const auto *output_impl = detail::requireDenseImpl(output, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  detail::transfer::requireExecution(output_impl, Execution::Cpu, kOpName,
                                     "output");

  const auto input_exec = input_impl->execution();
  if (input_exec != Execution::Mps && input_exec != Execution::Cuda) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyDeviceToHost: input must be a device tensor");
  }
  detail::transfer::requireMatchingShapeAndDType(output_impl, input_impl,
                                                 kOpName);
  detail::transfer::ensureRankSupported(input_impl, kOpName);
  detail::transfer::ensureRankSupported(output_impl, kOpName);

  const auto input_stats =
      detail::transfer::validateViewBounds(input_impl, kOpName, "input");
  const auto output_stats =
      detail::transfer::validateViewBounds(output_impl, kOpName, "output");

  if (input_stats.numel != output_stats.numel) {
    error::throwError(error::OrteafErrc::InvalidParameter,
                      "copyDeviceToHost: input/output numel must match");
  }
  if (input_stats.has_zero) {
    return;
  }

  dispatchTransferKernel(kOpName, Op::CopyDeviceToHost, input_exec, input_impl,
                         output_impl, "copyDeviceToHost kernel not found",
                         "copyDeviceToHost kernel execution failed");
}

} // namespace orteaf::extension::ops::dense
