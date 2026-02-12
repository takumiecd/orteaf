#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution/cpu/manager/cpu_kernel_base_manager.h"
#include "orteaf/internal/kernel/core/kernel_args.h"

#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h"
#endif
#if ORTEAF_ENABLE_CUDA
#include "orteaf/internal/execution/cuda/manager/cuda_kernel_base_manager.h"
#endif

namespace orteaf::internal::kernel::core {

/**
 * @brief Generic kernel entry structure.
 *
 * Holds a type-erased KernelBase lease. Each backend (CPU/MPS) provides
 * its own Lease type that implements a unified run(Args&) interface.
 */
class KernelEntry {
public:
  using Args = ::orteaf::internal::kernel::KernelArgs;

  using CpuKernelBaseLease = ::orteaf::internal::execution::cpu::manager::
      CpuKernelBaseManager::KernelBaseLease;

#if ORTEAF_ENABLE_MPS
  using MpsKernelBaseLease = ::orteaf::internal::execution::mps::manager::
      MpsKernelBaseManager::KernelBaseLease;
#endif

#if ORTEAF_ENABLE_CUDA
  using CudaKernelBaseLease = ::orteaf::internal::execution::cuda::manager::
      CudaKernelBaseManager::KernelBaseLease;
#endif

  using KernelBaseLease = std::variant<std::monostate, CpuKernelBaseLease
#if ORTEAF_ENABLE_MPS
                                       ,
                                       MpsKernelBaseLease
#endif
#if ORTEAF_ENABLE_CUDA
                                       ,
                                       CudaKernelBaseLease
#endif
                                       >;

  KernelEntry() = default;

  explicit KernelEntry(KernelBaseLease base) noexcept
      : base_(std::move(base)) {}

  KernelBaseLease &base() noexcept { return base_; }
  const KernelBaseLease &base() const noexcept { return base_; }

  void setBase(KernelBaseLease base) noexcept { base_ = std::move(base); }

  /**
   * @brief Run the kernel with unified interface.
   *
   * All backends implement run(Args&) on their Lease types.
   */
  void run(Args &args) {
    std::visit(
        [&](auto &v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Kernel base is not initialized");
          } else {
            // Unified call - all backends support v->run(args)
            if (!v) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Kernel base lease is invalid");
            }
            v->run(args);
          }
        },
        base_);
  }

private:
  /**
   * @brief Kernel base lease instance.
   */
  KernelBaseLease base_{};
};

} // namespace orteaf::internal::kernel::core
