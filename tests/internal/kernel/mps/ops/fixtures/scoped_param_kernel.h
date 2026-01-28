#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_entry.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

namespace orteaf::extension::kernel::mps::ops {

namespace kernel = ::orteaf::internal::kernel;
namespace mps_kernel = ::orteaf::internal::kernel::mps;

/**
 * @brief Parameter schema demonstrating storage-scoped params.
 *
 * NumElements is scoped to Input0 to show per-storage parameter binding.
 */
struct ScopedParamParams : kernel::ParamSchema<ScopedParamParams> {
  kernel::ScopedField<kernel::ParamId::NumElements, std::uint32_t,
                      kernel::OperandId::Input0>
      num_elements;

  ORTEAF_EXTRACT_FIELDS(num_elements)
};

/**
 * @brief Execute function for scoped-param example kernel.
 *
 * Extracts a scoped parameter and writes a global Count param as a side effect.
 */
inline void scopedParamExecute(mps_kernel::MpsKernelBase & /*base*/,
                               ::orteaf::internal::kernel::KernelArgs &args) {
  auto params = ScopedParamParams::extract(args);
  args.addParam(kernel::Param(kernel::ParamId::Count,
                              static_cast<std::size_t>(params.num_elements)));
}

/**
 * @brief Create and initialize a scoped-param kernel entry.
 */
inline mps_kernel::MpsKernelEntry createScopedParamKernel() {
  mps_kernel::MpsKernelEntry entry;
  entry.execute = scopedParamExecute;
  return entry;
}

} // namespace orteaf::extension::kernel::mps::ops

#endif // ORTEAF_ENABLE_MPS
