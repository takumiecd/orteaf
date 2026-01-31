#pragma once

namespace orteaf::internal::execution::cpu::resource {

/**
 * @brief Minimal CPU kernel base structure.
 *
 * Unlike GPU backends (MPS, CUDA), CPU kernels don't need to cache
 * any resources like pipeline states or modules. The kernel function
 * is directly callable, so this structure is intentionally empty.
 */
struct CpuKernelBase {
  CpuKernelBase() = default;
  
  CpuKernelBase(const CpuKernelBase &) = delete;
  CpuKernelBase &operator=(const CpuKernelBase &) = delete;
  CpuKernelBase(CpuKernelBase &&) = default;
  CpuKernelBase &operator=(CpuKernelBase &&) = default;
  ~CpuKernelBase() = default;
};

} // namespace orteaf::internal::execution::cpu::resource
