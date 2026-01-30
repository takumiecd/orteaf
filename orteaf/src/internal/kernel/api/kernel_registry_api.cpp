#include "orteaf/internal/kernel/api/kernel_registry_api.h"

namespace orteaf::internal::kernel::api {

::orteaf::internal::kernel::registry::KernelRegistry &kernelRegistry() noexcept {
  static ::orteaf::internal::kernel::registry::KernelRegistry instance;
  return instance;
}

} // namespace orteaf::internal::kernel::api
