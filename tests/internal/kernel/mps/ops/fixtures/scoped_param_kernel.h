#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/execution/cpu/resource/cpu_kernel_base.h>
#include <orteaf/internal/kernel/core/kernel_metadata.h>
#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

namespace orteaf::extension::kernel::mps::ops {

namespace kernel = ::orteaf::internal::kernel;

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
inline void scopedParamExecute(
    ::orteaf::internal::execution::cpu::resource::CpuKernelBase & /*base*/,
    ::orteaf::internal::kernel::KernelArgs &args) {
  auto params = ScopedParamParams::extract(args);
  args.addParam(kernel::Param(kernel::ParamId::Count,
                              static_cast<std::size_t>(params.num_elements)));
}

/**
 * @brief Create metadata for the scoped-param kernel.
 */
inline kernel::core::KernelMetadataLease createScopedParamMetadata() {
  using CpuExecutionApi =
      ::orteaf::internal::execution::cpu::api::CpuExecutionApi;
  auto metadata_lease = CpuExecutionApi::acquireKernelMetadata(scopedParamExecute);
  return kernel::core::KernelMetadataLease{std::move(metadata_lease)};
}

} // namespace orteaf::extension::kernel::mps::ops

#endif // ORTEAF_ENABLE_MPS
