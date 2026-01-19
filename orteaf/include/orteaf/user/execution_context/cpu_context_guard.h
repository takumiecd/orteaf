#pragma once

/**
 * @file cpu_context_guard.h
 * @brief RAII guard for switching the CPU execution context.
 */

#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution_context/cpu/current_context.h"

namespace orteaf::user::execution_context {

/**
 * @brief RAII guard that sets the CPU execution context for its lifetime.
 *
 * Captures the current context on construction and restores it on destruction.
 * The current context is global (not thread-local).
 *
 * @par Usage (default device)
 * @code
 * #include <orteaf/user/execution_context/cpu_context_guard.h>
 *
 * using ::orteaf::user::execution_context::CpuExecutionContextGuard;
 *
 * void run_on_cpu() {
 *   CpuExecutionContextGuard guard; // uses CpuDeviceHandle{0}
 *   // CPU work here
 * }
 * @endcode
 *
 * @par Usage (explicit device handle)
 * @code
 * #include <orteaf/internal/execution/cpu/cpu_handles.h>
 * #include <orteaf/user/execution_context/cpu_context_guard.h>
 *
 * CpuExecutionContextGuard guard(
 *     ::orteaf::internal::execution::cpu::CpuDeviceHandle{0});
 * @endcode
 *
 * @note The CPU execution manager must be configured before creating the guard.
 */
class CpuExecutionContextGuard {
public:
  /// @brief Use the default CPU device (handle 0).
  CpuExecutionContextGuard();
  /// @brief Use the specified CPU device handle.
  explicit CpuExecutionContextGuard(
      ::orteaf::internal::execution::cpu::CpuDeviceHandle device);

  CpuExecutionContextGuard(const CpuExecutionContextGuard &) = delete;
  CpuExecutionContextGuard &operator=(const CpuExecutionContextGuard &) = delete;

  CpuExecutionContextGuard(CpuExecutionContextGuard &&other) noexcept;
  CpuExecutionContextGuard &operator=(CpuExecutionContextGuard &&other) noexcept;

  ~CpuExecutionContextGuard();

private:
  void activate(::orteaf::internal::execution_context::cpu::Context context);
  void release() noexcept;

  ::orteaf::internal::execution_context::cpu::CurrentContext previous_{};
  bool active_{false};
};

} // namespace orteaf::user::execution_context
