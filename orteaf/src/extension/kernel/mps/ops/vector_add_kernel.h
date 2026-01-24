#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/execution_context/mps/context.h>
#include <orteaf/internal/kernel/kernel_param_schema.h>
#include <orteaf/internal/kernel/kernel_storage_schema.h>
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_entry.h>
#include <orteaf/internal/kernel/mps/mps_storage_binding.h>
#include <orteaf/internal/kernel/param_id.h>
#include <orteaf/internal/kernel/storage_id.h>

namespace orteaf::extension::kernel::mps::ops {

namespace kernel = ::orteaf::internal::kernel;
namespace mps_kernel = ::orteaf::internal::kernel::mps;

/**
 * @brief Storage schema for vector add kernel.
 *
 * Defines the required buffer bindings:
 * - Input0: First input vector (A)
 * - Input1: Second input vector (B)
 * - Output: Output vector (C = A + B)
 */
struct VectorAddStorages : kernel::StorageSchema<VectorAddStorages> {
  kernel::StorageField<kernel::StorageId::Input0> a;
  kernel::StorageField<kernel::StorageId::Input1> b;
  kernel::StorageField<kernel::StorageId::Output> c;

  ORTEAF_EXTRACT_STORAGES(a, b, c)
};

/**
 * @brief Parameter schema for vector add kernel.
 *
 * Defines the required parameters:
 * - NumElements: Number of elements to process
 */
struct VectorAddParams : kernel::ParamSchema<VectorAddParams> {
  kernel::Field<kernel::ParamId::NumElements, std::size_t> num_elements;

  ORTEAF_EXTRACT_FIELDS(num_elements)
};

/**
 * @brief Execute function for vector add kernel.
 *
 * Encodes and dispatches the vector add compute shader.
 * This function is called by MpsKernelEntry::run().
 *
 * @param base Configured MpsKernelBase with cached pipeline state
 * @param args Kernel arguments containing storages and parameters
 */
inline void vectorAddExecute(mps_kernel::MpsKernelBase &base,
                             mps_kernel::MpsKernelArgs &args) {
  // Extract storages and params from args using schemas
  auto storages = VectorAddStorages::extract(args);
  auto params = VectorAddParams::extract(args);

  // Create command buffer and encoder
  auto command_buffer = base.createCommandBuffer(args.context());
  if (!command_buffer) {
    return;
  }

  auto encoder = base.createComputeCommandEncoder(command_buffer);
  if (!encoder) {
    return;
  }

  // Wait for storage dependencies (RAW hazards)
  base.waitAllStorageDependencies<mps_kernel::MpsStorageBinding>(
      encoder, storages.a, storages.b, storages.c);

  // Get pipeline state and set it on encoder
  auto *pipeline = base.getPipeline(args.context().device.payloadHandle(), 0);
  if (!pipeline) {
    base.endEncoding(encoder);
    return;
  }
  base.setPipelineState(encoder, *pipeline);

  // Bind buffers using storage schema (indices match Metal shader)
  // buffer(0) = a, buffer(1) = b, buffer(2) = c
  storages.bindAll(base, encoder, 0);

  // Bind parameters (index 3 = num_elements)
  params.bindAll(base, encoder, 3);

  // Calculate thread dimensions
  const std::size_t num_elements = params.num_elements;
  constexpr std::size_t kThreadsPerThreadgroup = 256;

  auto threads_per_grid = mps_kernel::MpsKernelBase::makeGridSize(num_elements);
  auto threads_per_threadgroup =
      mps_kernel::MpsKernelBase::makeThreadsPerThreadgroup(
          kThreadsPerThreadgroup);

  // Dispatch threads
  base.dispatchThreads(encoder, threads_per_grid, threads_per_threadgroup);

  // Update storage tokens for WAW/WAR hazards
  base.updateAllStorageTokens<mps_kernel::MpsStorageBinding>(
      args.context(), command_buffer, encoder, storages.a, storages.b,
      storages.c);

  // End encoding and commit
  base.endEncoding(encoder);
  base.commit(command_buffer);
}

/**
 * @brief Create and initialize a vector add kernel entry.
 *
 * This factory function creates an MpsKernelEntry configured for vector add.
 *
 * @return MpsKernelEntry for vector add operations
 */
inline mps_kernel::MpsKernelEntry createVectorAddKernel() {
  mps_kernel::MpsKernelEntry entry;

  // Register the Metal library and function
  // Library name should match the embedded library name
  entry.base.addKey("vector_add", "orteaf_vector_add");

  // Set the execute function
  entry.execute = vectorAddExecute;

  return entry;
}

} // namespace orteaf::extension::kernel::mps::ops

#endif // ORTEAF_ENABLE_MPS
