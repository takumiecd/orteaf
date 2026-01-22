#include "orteaf/internal/kernel/cpu/cpu_kernel_args.h"

#include "orteaf/internal/execution_context/cpu/current_context.h"

namespace orteaf::internal::kernel::cpu {

CpuKernelArgs CpuKernelArgs::fromCurrentContext() {
  return CpuKernelArgs(
      ::orteaf::internal::execution_context::cpu::currentContext());
}

CpuKernelArgs::CpuKernelArgs()
    : context_(::orteaf::internal::execution_context::cpu::currentContext()) {}

CpuKernelArgs::CpuKernelArgs(Context context) : context_(std::move(context)) {}

} // namespace orteaf::internal::kernel::cpu
