#include "orteaf/internal/execution/cuda/resource/cuda_kernel_base.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/execution_context/cuda/context.h"
#include "orteaf/internal/kernel/core/kernel_args.h"

namespace orteaf::internal::execution::cuda::resource {

void CudaKernelBase::configureFunctions(ContextLease &context_lease) {
  const auto context = context_lease.payloadHandle();
  auto *context_resource = context_lease.operator->();
  if (context_resource == nullptr) {
    return;
  }

  auto entry_idx = findContextIndex(context);
  if (entry_idx == kInvalidIndex) {
    ContextFunctions entry{};
    entry.context = context;
    context_functions_.pushBack(std::move(entry));
    entry_idx = context_functions_.size() - 1;
  }

  auto &entry = context_functions_[entry_idx];
  entry.modules.clear();
  entry.functions.clear();
  entry.modules.reserve(keys_.size());
  entry.functions.reserve(keys_.size());

  for (std::size_t i = 0; i < keys_.size(); ++i) {
    const auto &key = keys_[i];
    auto module_lease = context_resource->module_manager.acquire(key.first);
    FunctionLease function_lease{};
    auto *module_resource = module_lease.operator->();
    if (module_resource != nullptr) {
      function_lease = module_resource->function_manager.acquire(key.second);
    }
    entry.modules.pushBack(std::move(module_lease));
    entry.functions.pushBack(std::move(function_lease));
  }

  entry.configured = true;
  for (const auto &function : entry.functions) {
    if (!function) {
      entry.configured = false;
      break;
    }
  }
}

bool CudaKernelBase::setKeys(
    const ::orteaf::internal::base::HeapVector<Key> &keys) {
  reset();
  keys_.reserve(keys.size());
  for (const auto &key : keys) {
    keys_.pushBack(key);
  }
  return true;
}

bool CudaKernelBase::ensureFunctions(ContextLease &context_lease) {
  if (!context_lease) {
    return false;
  }
  if (keys_.empty()) {
    return true;
  }

  const auto context = context_lease.payloadHandle();
  if (!configured(context)) {
    configureFunctions(context_lease);
  }

  const auto idx = findContextIndex(context);
  if (idx == kInvalidIndex) {
    return false;
  }

  const auto &entry = context_functions_[idx];
  if (!entry.configured || entry.functions.size() != keys_.size()) {
    return false;
  }
  for (const auto &function : entry.functions) {
    if (!function) {
      return false;
    }
  }
  return true;
}

void CudaKernelBase::run(::orteaf::internal::kernel::KernelArgs &args) {
  auto *context =
      args.context().tryAs<::orteaf::internal::execution_context::cuda::Context>();
  if (context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidParameter,
        "CUDA kernel requires CUDA execution context");
  }
  if (!ensureFunctions(context->context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "CUDA kernel base could not be initialized");
  }
  if (!execute_) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Kernel execute function is invalid");
  }
  execute_(*this, args);
}

} // namespace orteaf::internal::execution::cuda::resource

#endif // ORTEAF_ENABLE_CUDA
