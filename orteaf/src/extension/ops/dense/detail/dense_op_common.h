#pragma once

#include <cstdint>
#include <string>

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/architecture/architecture.h>
#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/core/context_any.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/key_components.h>
#include <orteaf/internal/kernel/dispatch/dispatcher.h>
#include <orteaf/user/tensor/tensor.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution_context/mps/current_context.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution_context/cuda/current_context.h>
#endif

namespace orteaf::extension::ops::dense::detail {

using Tensor = ::orteaf::user::tensor::Tensor;
using DenseTensorImpl = ::orteaf::extension::tensor::DenseTensorImpl;
namespace error = ::orteaf::internal::diagnostics::error;
namespace kernel = ::orteaf::internal::kernel;
using Execution = ::orteaf::internal::execution::Execution;
using Architecture = ::orteaf::internal::architecture::Architecture;

enum class ExecutionSupport : std::uint8_t {
  CpuOnly,
  CpuOrMps,
  CpuOrCuda,
  CpuOrMpsOrCuda,
};

inline void ensureValidTensor(const Tensor &tensor, const char *op_name) {
  if (!tensor.valid()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) + ": tensor is not valid");
  }
}

inline const DenseTensorImpl *requireDenseImpl(const Tensor &tensor,
                                               const char *op_name) {
  const auto *lease = tensor.tryAs<DenseTensorImpl>();
  if (!lease || !(*lease) || lease->operator->() == nullptr) {
    error::throwError(error::OrteafErrc::Unsupported,
                      std::string(op_name) + ": dense tensor is required");
  }
  return lease->operator->();
}

inline bool allowsMps(ExecutionSupport support) noexcept {
  return support == ExecutionSupport::CpuOrMps ||
         support == ExecutionSupport::CpuOrMpsOrCuda;
}

inline bool allowsCuda(ExecutionSupport support) noexcept {
  return support == ExecutionSupport::CpuOrCuda ||
         support == ExecutionSupport::CpuOrMpsOrCuda;
}

inline kernel::KernelArgs makeArgsForExecution(Execution execution,
                                               ExecutionSupport support,
                                               const char *op_name) {
  switch (execution) {
  case Execution::Cpu:
    return kernel::KernelArgs(kernel::ContextAny::erase(
        ::orteaf::internal::execution_context::cpu::currentContext()));
  case Execution::Mps:
    if (!allowsMps(support)) {
      break;
    }
#if ORTEAF_ENABLE_MPS
    return kernel::KernelArgs(kernel::ContextAny::erase(
        ::orteaf::internal::execution_context::mps::currentContext()));
#else
    error::throwError(error::OrteafErrc::ExecutionUnavailable,
                      std::string(op_name) +
                          ": built without MPS support");
#endif
  case Execution::Cuda:
    if (!allowsCuda(support)) {
      break;
    }
#if ORTEAF_ENABLE_CUDA
    return kernel::KernelArgs(kernel::ContextAny::erase(
        ::orteaf::internal::execution_context::cuda::currentContext()));
#else
    error::throwError(error::OrteafErrc::ExecutionUnavailable,
                      std::string(op_name) +
                          ": built without CUDA support");
#endif
  default:
    break;
  }

  error::throwError(error::OrteafErrc::ExecutionUnavailable,
                    std::string(op_name) +
                        ": execution is not supported");
}

inline kernel::KernelArgs makeArgsForCpuOnly(Execution execution,
                                             const char *op_name) {
  return makeArgsForExecution(execution, ExecutionSupport::CpuOnly, op_name);
}

inline kernel::KernelArgs makeArgsForCpuOrMps(Execution execution,
                                              const char *op_name) {
  return makeArgsForExecution(execution, ExecutionSupport::CpuOrMps, op_name);
}

inline kernel::KernelArgs makeArgsForCpuOrCuda(Execution execution,
                                               const char *op_name) {
  return makeArgsForExecution(execution, ExecutionSupport::CpuOrCuda, op_name);
}

inline kernel::KernelArgs makeArgsForCpuOrMpsOrCuda(Execution execution,
                                                    const char *op_name) {
  return makeArgsForExecution(execution, ExecutionSupport::CpuOrMpsOrCuda,
                              op_name);
}

inline Architecture architectureForArgs(const kernel::KernelArgs &args,
                                        const char *op_name) {
  if (!args.valid()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": kernel args context is not valid");
  }
  const auto arch = args.architecture();
  if (::orteaf::internal::architecture::executionOf(arch) != args.execution()) {
    error::throwError(error::OrteafErrc::InvalidState,
                      std::string(op_name) +
                          ": context architecture does not match execution");
  }
  return arch;
}

inline void dispatchOrThrow(const kernel::KeyRequest &request,
                            kernel::KernelArgs &args,
                            const char *not_found_message,
                            const char *failed_message) {
  kernel::dispatch::Dispatcher dispatcher;
  const auto result = dispatcher.dispatch(request, args);
  if (result.notFound()) {
    error::throwError(error::OrteafErrc::OperationFailed, not_found_message);
  }
  if (result.failed()) {
    error::throwError(error::OrteafErrc::OperationFailed, failed_message);
  }
}

[[noreturn]] inline void throwExecutionUnavailable(const char *op_name,
                                                   const char *backend_name) {
  error::throwError(error::OrteafErrc::ExecutionUnavailable,
                    std::string(op_name) + ": built without " + backend_name +
                        " support");
}

[[noreturn]] inline void throwMpsUnavailable(const char *op_name) {
  throwExecutionUnavailable(op_name, "MPS");
}

} // namespace orteaf::extension::ops::dense::detail
