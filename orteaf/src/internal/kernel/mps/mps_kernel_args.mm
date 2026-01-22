#include "orteaf/internal/kernel/mps/mps_kernel_args.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution_context/mps/current_context.h"

namespace orteaf::internal::kernel::mps {

MpsKernelArgs MpsKernelArgs::fromCurrentContext() {
  return MpsKernelArgs(
      ::orteaf::internal::execution_context::mps::currentContext());
}

MpsKernelArgs::MpsKernelArgs()
    : context_(::orteaf::internal::execution_context::mps::currentContext()) {}

MpsKernelArgs::MpsKernelArgs(Context context) : context_(std::move(context)) {}

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
