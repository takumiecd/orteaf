#pragma once

#include <optional>

#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/core/kernel_entry.h"
#include "orteaf/internal/kernel/core/key_components.h"

namespace orteaf::internal::kernel::dispatch {

/**
 * @brief Result of a kernel dispatch operation.
 */
enum class DispatchStatus {
  Success,        ///< Kernel found and executed successfully
  NotFound,       ///< No matching kernel found
  ExecutionError, ///< Kernel found but execution failed
};

/**
 * @brief Result wrapper for dispatch operations.
 */
struct DispatchResult {
  DispatchStatus status;
  
  [[nodiscard]] bool success() const noexcept {
    return status == DispatchStatus::Success;
  }
  
  [[nodiscard]] bool notFound() const noexcept {
    return status == DispatchStatus::NotFound;
  }
  
  [[nodiscard]] bool failed() const noexcept {
    return status == DispatchStatus::ExecutionError;
  }
};

/**
 * @brief Dispatcher for kernel execution.
 *
 * Coordinates between KeyResolver and KernelRegistry to find and execute
 * the appropriate kernel for a given operation.
 *
 * Workflow:
 * 1. Operation creates KernelArgs
 * 2. Operation provides KeyRequest
 * 3. Dispatcher uses KeyResolver to resolve the key
 * 4. Dispatcher looks up kernel in KernelRegistry
 * 5. Dispatcher executes the kernel
 */
class Dispatcher {
public:
  using Entry = ::orteaf::internal::kernel::core::KernelEntry;
  using Args = ::orteaf::internal::kernel::KernelArgs;
  using Request = ::orteaf::internal::kernel::KeyRequest;
  
  Dispatcher() = default;
  
  /**
   * @brief Dispatch and execute a kernel.
   *
   * Resolves the kernel key from the request, looks it up in the registry,
   * and executes it with the provided arguments.
   *
   * @param request The key request (Op, DType, Architecture)
   * @param args The kernel arguments
   * @return Dispatch result indicating success or failure reason
   */
  DispatchResult dispatch(const Request &request, Args &args);
  
  /**
   * @brief Resolve a kernel without executing it.
   *
   * Useful for checking if a kernel exists before committing to execution.
   *
   * @param request The key request (Op, DType, Architecture)
   * @param args The kernel arguments (for verification)
   * @return Pointer to the kernel entry if found, nullptr otherwise
   */
  Entry *resolve(const Request &request, const Args &args);
};

} // namespace orteaf::internal::kernel::dispatch
