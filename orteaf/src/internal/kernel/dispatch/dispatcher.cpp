#include "orteaf/internal/kernel/dispatch/dispatcher.h"

#include "orteaf/internal/kernel/api/kernel_registry_api.h"
#include "orteaf/internal/kernel/core/key_resolver.h"

namespace orteaf::internal::kernel::dispatch {

DispatchResult Dispatcher::dispatch(const Request &request, Args &args) {
  // Resolve the kernel key
  auto *entry = resolve(request, args);
  
  if (!entry) {
    return DispatchResult{DispatchStatus::NotFound};
  }
  
  // Execute the kernel
  try {
    entry->run(args);
    return DispatchResult{DispatchStatus::Success};
  } catch (...) {
    return DispatchResult{DispatchStatus::ExecutionError};
  }
}

Dispatcher::Entry *Dispatcher::resolve(const Request &request, const Args &args) {
  // Get the global kernel registry
  auto &registry = ::orteaf::internal::kernel::api::KernelRegistryApi::instance();
  
  // Use key resolver to find the best matching kernel key
  auto key_opt = ::orteaf::internal::kernel::key_resolver::resolve(
      registry, request, args);
  
  if (!key_opt.has_value()) {
    return nullptr;
  }
  
  // Look up the kernel in the registry
  return registry.lookup(key_opt.value());
}

} // namespace orteaf::internal::kernel::dispatch
