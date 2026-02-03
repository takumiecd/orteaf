#pragma once

#include <cstddef>

namespace orteaf::internal::kernel::registry {

using RegisterFn = void (*)();

bool addKernelRegistrar(RegisterFn fn);

void registerAllKernels();

}  // namespace orteaf::internal::kernel::registry

#define ORTEAF_INTERNAL_CONCAT_IMPL(a, b) a##b
#define ORTEAF_INTERNAL_CONCAT(a, b) ORTEAF_INTERNAL_CONCAT_IMPL(a, b)

#ifdef __COUNTER__
#define ORTEAF_INTERNAL_UNIQUE_ID(base) ORTEAF_INTERNAL_CONCAT(base, __COUNTER__)
#else
#define ORTEAF_INTERNAL_UNIQUE_ID(base) ORTEAF_INTERNAL_CONCAT(base, __LINE__)
#endif

#define ORTEAF_REGISTER_KERNEL(fn)                                             \
  namespace {                                                                  \
  const bool ORTEAF_INTERNAL_UNIQUE_ID(_orteaf_kernel_reg_) =                  \
      ::orteaf::internal::kernel::registry::addKernelRegistrar(fn);            \
  }
