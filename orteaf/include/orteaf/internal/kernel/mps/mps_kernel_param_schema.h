#pragma once

#if ORTEAF_ENABLE_MPS

// Import generic kernel parameter schema
#include <orteaf/internal/kernel/kernel_param_schema.h>
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>

namespace orteaf::internal::kernel::mps {

// Re-export generic types in mps namespace for backward compatibility
using ::orteaf::internal::kernel::Field;
using ::orteaf::internal::kernel::OptionalField;
using ::orteaf::internal::kernel::ParamSchema;

} // namespace orteaf::internal::kernel::mps

#endif // ORTEAF_ENABLE_MPS
