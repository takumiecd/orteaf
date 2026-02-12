#pragma once

#include "orteaf/internal/execution/cpu/resource/cpu_kernel_base.h"

namespace orteaf::internal::execution::cpu::resource {

/**
 * @brief Kernel metadata resource for CPU.
 *
 * Stores ExecuteFunc for kernel reconstruction.
 */
struct CpuKernelMetadata {
  using ExecuteFunc =
      ::orteaf::internal::execution::cpu::resource::CpuKernelBase::ExecuteFunc;

  CpuKernelMetadata() = default;

  ExecuteFunc execute() const noexcept { return execute_; }
  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

private:
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::execution::cpu::resource
