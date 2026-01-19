#pragma once

/**
 * @file mps_context_guard.h
 * @brief RAII guard for switching the MPS execution context.
 */

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution_context/mps/current_context.h"

namespace orteaf::user::execution_context {

/**
 * @brief RAII guard that sets the MPS execution context for its lifetime.
 *
 * Captures the current context on construction and restores it on destruction.
 * The current context is global (not thread-local).
 *
 * @par Usage (default device + new command queue)
 * @code
 * #include <orteaf/user/execution_context/mps_context_guard.h>
 *
 * using ::orteaf::user::execution_context::MpsExecutionContextGuard;
 *
 * void run_on_mps() {
 *   MpsExecutionContextGuard guard; // uses MpsDeviceHandle{0}
 *   // MPS work here
 * }
 * @endcode
 *
 * @par Usage (explicit device + new command queue)
 * @code
 * #include <orteaf/internal/execution/mps/mps_handles.h>
 * #include <orteaf/user/execution_context/mps_context_guard.h>
 *
 * MpsExecutionContextGuard guard(
 *     ::orteaf::internal::execution::mps::MpsDeviceHandle{0});
 * @endcode
 *
 * @par Usage (explicit device + explicit command queue)
 * @code
 * #include <orteaf/internal/execution/mps/mps_handles.h>
 * #include <orteaf/user/execution_context/mps_context_guard.h>
 *
 * MpsExecutionContextGuard guard(
 *     ::orteaf::internal::execution::mps::MpsDeviceHandle{0},
 *     ::orteaf::internal::execution::mps::MpsCommandQueueHandle{1});
 * @endcode
 *
 * @note The MPS execution manager must be configured before creating the guard.
 */
class MpsExecutionContextGuard {
public:
  /// @brief Use the default MPS device (handle 0) and a new command queue.
  MpsExecutionContextGuard();
  /// @brief Use the specified MPS device and a new command queue.
  explicit MpsExecutionContextGuard(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device);
  /// @brief Use the specified MPS device and command queue handles.
  MpsExecutionContextGuard(
      ::orteaf::internal::execution::mps::MpsDeviceHandle device,
      ::orteaf::internal::execution::mps::MpsCommandQueueHandle command_queue);

  MpsExecutionContextGuard(const MpsExecutionContextGuard &) = delete;
  MpsExecutionContextGuard &operator=(const MpsExecutionContextGuard &) = delete;

  MpsExecutionContextGuard(MpsExecutionContextGuard &&other) noexcept;
  MpsExecutionContextGuard &operator=(MpsExecutionContextGuard &&other) noexcept;

  ~MpsExecutionContextGuard();

private:
  void activate(::orteaf::internal::execution_context::mps::Context context);
  void release() noexcept;

  ::orteaf::internal::execution_context::mps::CurrentContext previous_{};
  bool active_{false};
};

} // namespace orteaf::user::execution_context

#endif // ORTEAF_ENABLE_MPS
