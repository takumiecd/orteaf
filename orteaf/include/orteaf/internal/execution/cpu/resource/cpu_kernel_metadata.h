#pragma once

#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/execution/cpu/resource/cpu_kernel_base.h"

namespace orteaf::internal::execution::cpu::resource {

/**
 * @brief Kernel metadata resource for CPU.
 *
 * CPU kernels don't need to store any metadata for reconstruction
 * since the kernel function is directly callable and doesn't require
 * any cached resources like GPU backends do.
 *
 * This structure is intentionally empty, matching CpuKernelBase.
 */
struct CpuKernelMetadata {
  CpuKernelMetadata() = default;
};

} // namespace orteaf::internal::execution::cpu::resource
