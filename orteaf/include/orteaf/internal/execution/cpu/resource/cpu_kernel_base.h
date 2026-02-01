#pragma once

namespace orteaf::internal::kernel {
class KernelArgs;
} // namespace orteaf::internal::kernel

namespace orteaf::internal::kernel::core {
class KernelEntry;
} // namespace orteaf::internal::kernel::core

namespace orteaf::internal::execution::cpu::resource {
struct CpuKernelMetadata;
} // namespace orteaf::internal::execution::cpu::resource

namespace orteaf::internal::execution::cpu::manager {
struct KernelBasePayloadPoolTraits;
} // namespace orteaf::internal::execution::cpu::manager

namespace orteaf::internal::execution::cpu::resource {

/**
 * @brief CPU kernel base structure with ExecuteFunc and run() logic.
 *
 * Unlike GPU backends (MPS, CUDA), CPU kernels don't need to cache
 * resources like pipeline states. However, this structure holds the
 * ExecuteFunc for type-safe kernel execution.
 */
struct CpuKernelBase {
  using MetadataType =
      ::orteaf::internal::execution::cpu::resource::CpuKernelMetadata;
  using ExecuteFunc = void (*)(CpuKernelBase &,
                               ::orteaf::internal::kernel::KernelArgs &);

  CpuKernelBase() = default;

  CpuKernelBase(const CpuKernelBase &) = delete;
  CpuKernelBase &operator=(const CpuKernelBase &) = delete;
  CpuKernelBase(CpuKernelBase &&) = default;
  CpuKernelBase &operator=(CpuKernelBase &&) = default;
  ~CpuKernelBase() = default;

  // Public getter for ExecuteFunc (needed by Metadata)
  ExecuteFunc execute() const noexcept { return execute_; }

private:
  friend class ::orteaf::internal::kernel::core::KernelEntry;
  friend struct ::orteaf::internal::execution::cpu::manager::
      KernelBasePayloadPoolTraits;

  void run(::orteaf::internal::kernel::KernelArgs &args);

  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::execution::cpu::resource
