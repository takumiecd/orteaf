#include <orteaf/extension/ops/dense/dense_tensor_ops.h>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

#include "detail/dense_op_common.h"

namespace orteaf::extension::ops::dense {

namespace {

namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;

} // namespace

void DenseTensorOps::print(const Tensor &input) {
  constexpr const char *kOpName = "print";
  detail::ensureValidTensor(input, kOpName);
  const auto *input_impl = detail::requireDenseImpl(input, kOpName);

  if (!input_impl->isContiguous() || input_impl->offset() != 0) {
    error::throwError(error::OrteafErrc::Unsupported,
                      "Print op currently requires contiguous tensors");
  }

  auto args = detail::makeArgsForCpuOnly(input_impl->execution(), kOpName);
  input_impl->bindAllArgs(args, kernel::OperandId::Input0);

  const kernel::KeyRequest request{::orteaf::internal::ops::Op::Print,
                                   input_impl->dtype(),
                                   detail::architectureForArgs(args, kOpName)};

  detail::dispatchOrThrow(request, args,
                          "Print kernel not found for requested dtype",
                          "Print kernel execution failed");
}

} // namespace orteaf::extension::ops::dense
