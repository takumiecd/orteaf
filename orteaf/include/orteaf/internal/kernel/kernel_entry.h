#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/kernel/core/kernel_args.h"

#if ORTEAF_ENABLE_MPS
#include "orteaf/internal/execution/mps/manager/mps_kernel_base_manager.h"
#include "orteaf/internal/execution_context/mps/context.h"
#endif

namespace orteaf::internal::kernel {

/**
 * @brief Generic kernel entry structure.
 *
 * Holds a type-erased KernelBase and backend-specific execution function.
 */
class KernelEntry {
public:
  using Args = ::orteaf::internal::kernel::KernelArgs;

#if ORTEAF_ENABLE_MPS
  using MpsKernelBaseLease =
      ::orteaf::internal::execution::mps::manager::MpsKernelBaseManager::
          KernelBaseLease;
#endif

  using KernelBaseLease = std::variant<
      std::monostate
#if ORTEAF_ENABLE_MPS
      ,
      MpsKernelBaseLease
#endif
      >;

  using ExecuteFunc = void (*)(KernelBaseLease &lease, Args &args);

  KernelEntry() = default;

  explicit KernelEntry(KernelBaseLease base, ExecuteFunc execute) noexcept
      : base_(std::move(base)), execute_(execute) {}

  KernelBaseLease &base() noexcept { return base_; }
  const KernelBaseLease &base() const noexcept { return base_; }

  void setBase(KernelBaseLease base) noexcept { base_ = std::move(base); }

  ExecuteFunc execute() const noexcept { return execute_; }
  void setExecute(ExecuteFunc execute) noexcept { execute_ = execute; }

  /**
   * @brief Run the kernel with automatic configuration.
   */
  void run(Args &args) {
    std::visit(
        [&](auto &lease_value) {
          using LeaseT = std::decay_t<decltype(lease_value)>;
          if constexpr (std::is_same_v<LeaseT, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Kernel base is not initialized");
#if ORTEAF_ENABLE_MPS
          } else if constexpr (std::is_same_v<LeaseT, MpsKernelBaseLease>) {
            auto *base_ptr = lease_value.operator->();
            if (!base_ptr) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "MPS kernel base lease is invalid");
            }
            auto *context = args.context()
                                .tryAs<
                                    ::orteaf::internal::execution_context::mps::
                                        Context>();
            if (!context) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidParameter,
                  "MPS kernel requires MPS execution context");
            }
            auto device = context->device.payloadHandle();
            if (!base_ptr->configured(device)) {
              base_ptr->configure(context->device);
            }
            if (!execute_) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Kernel execute function is invalid");
            }
            execute_(base_, args);
#endif
          }
        },
        base_);
  }

private:
  /**
   * @brief Kernel base instance.
   */
  KernelBaseLease base_{};

  /**
   * @brief Execution function pointer.
   */
  ExecuteFunc execute_{nullptr};
};

} // namespace orteaf::internal::kernel
