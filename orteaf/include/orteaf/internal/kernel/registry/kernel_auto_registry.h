#pragma once

#include <cstddef>

namespace orteaf::internal::kernel::registry {

using RegisterFn = void (*)();

bool addKernelRegistrar(RegisterFn fn);

void registerAllKernels();

}  // namespace orteaf::internal::kernel::registry

#define ORTEAF_INTERNAL_CONCAT_IMPL(a, b) a##b
#define ORTEAF_INTERNAL_CONCAT(a, b) ORTEAF_INTERNAL_CONCAT_IMPL(a, b)

#define ORTEAF_REGISTER_KERNEL(fn)                                             \
  namespace {                                                                  \
  const bool ORTEAF_INTERNAL_CONCAT(_orteaf_kernel_reg_, __LINE__) =            \
      ::orteaf::internal::kernel::registry::addKernelRegistrar(fn);            \
  }
