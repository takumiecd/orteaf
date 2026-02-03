#include "orteaf/internal/kernel/registry/kernel_auto_registry.h"

#include <orteaf/internal/base/small_vector.h>

namespace orteaf::internal::kernel::registry {

namespace {

using Registry = ::orteaf::internal::base::SmallVector<RegisterFn, 32>;

Registry &registrars() {
  static Registry registry;
  return registry;
}

}  // namespace

bool addKernelRegistrar(RegisterFn fn) {
  if (!fn) {
    return false;
  }
  auto &registry = registrars();
  for (auto existing : registry) {
    if (existing == fn) {
      return false;
    }
  }
  registry.pushBack(fn);
  return true;
}

void registerAllKernels() {
  auto &registry = registrars();
  for (auto fn : registry) {
    fn();
  }
}

}  // namespace orteaf::internal::kernel::registry
