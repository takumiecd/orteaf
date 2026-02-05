#include "orteaf/internal/kernel/registry/kernel_auto_registry.h"

#include "orteaf/internal/kernel/registry/kernel_generated_registry.h"

namespace orteaf::internal::kernel::registry {

bool addKernelRegistrar(RegisterFn /*fn*/) {
  // Dynamic registrars are currently disabled.
  return false;
}

void registerAllKernels() {
  registerAllGeneratedKernels();
}

}  // namespace orteaf::internal::kernel::registry
