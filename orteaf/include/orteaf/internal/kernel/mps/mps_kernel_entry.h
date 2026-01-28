#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/kernel/core/kernel_args.h"
#include "orteaf/internal/kernel/mps/mps_kernel_base.h"

namespace orteaf::internal::kernel::mps {

/**
 * @brief MPS kernel entry structure.
 *
 * Holds a KernelBase instance, execution function, and kernel identifier.
 * Provides automatic configure and execution via run() method.
 *
 * The KernelBase is directly owned - resource sharing happens at the
 * PipelineLease level within MpsComputePipelineStateManager.
 */
struct MpsKernelEntry {
  using ExecuteFunc =
      void (*)(MpsKernelBase &base, ::orteaf::internal::kernel::KernelArgs &args);

  /**
   * @brief Kernel base instance.
   *
   * Caches MTLComputePipelineState per device. The underlying PipelineLeases
   * are shared across all KernelBase instances via MpsComputePipelineStateManager.
   */
  MpsKernelBase base;

  /**
   * @brief Execution function pointer.
   *
   * This function is called by run() with the configured KernelBase
   * and kernel arguments.
   */
  ExecuteFunc execute{nullptr};

  /**
   * @brief Run the kernel with automatic configuration.
   *
   * If the KernelBase is not configured for the device in args.context(),
   * it will be configured before execution.
   *
   * @param args Kernel arguments containing context, storages, and parameters
   */
  void run(::orteaf::internal::kernel::KernelArgs &args) {
    auto *context =
        args.context().tryAs<::orteaf::internal::execution_context::mps::Context>();
    if (!context) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
          "MPS kernel requires MPS execution context");
    }
    auto device = context->device.payloadHandle();
    if (!base.configured(device)) {
      base.configure(*context);
    }
    execute(base, args);
  }
};

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
