#pragma once

#include <cstddef>

namespace orteaf::internal::kernel::registry {

// Register kernels discovered by the generated registry.
void registerAllKernels();

}  // namespace orteaf::internal::kernel::registry

#define ORTEAF_INTERNAL_CONCAT_IMPL(a, b) a##b
#define ORTEAF_INTERNAL_CONCAT(a, b) ORTEAF_INTERNAL_CONCAT_IMPL(a, b)

#ifdef __COUNTER__
#define ORTEAF_INTERNAL_UNIQUE_ID(base) ORTEAF_INTERNAL_CONCAT(base, __COUNTER__)
#else
#define ORTEAF_INTERNAL_UNIQUE_ID(base) ORTEAF_INTERNAL_CONCAT(base, __LINE__)
#endif

// Marker macro for the generated kernel registry (no-op at runtime).
#define ORTEAF_REGISTER_KERNEL(fn) /* used by codegen only */
