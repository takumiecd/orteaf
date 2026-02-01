#include "orteaf/internal/kernel/api/kernel_registry_api.h"

namespace orteaf::internal::kernel::api {

KernelRegistryApi::Registry &KernelRegistryApi::instance() noexcept {
  static KernelRegistryApi::Registry registry;
  return registry;
}

} // namespace orteaf::internal::kernel::api
