#include "orteaf/internal/execution/cpu/resource/cpu_kernel_base.h"

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/kernel/core/kernel_args.h"

namespace orteaf::internal::execution::cpu::resource {

void CpuKernelBase::run(::orteaf::internal::kernel::KernelArgs &args) {
  if (!execute_) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Kernel execute function is invalid");
  }
  execute_(*this, args);
}

} // namespace orteaf::internal::execution::cpu::resource
