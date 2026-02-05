#include "orteaf/internal/kernel/registry/kernel_auto_registry.h"

#include "orteaf/internal/kernel/registry/kernel_generated_registry.h"

namespace orteaf::internal::kernel::registry {

void registerAllKernels() {
  registerAllGeneratedKernels();
}

}  // namespace orteaf::internal::kernel::registry
