#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/kernel/kernel_param_schema.h>
#include <orteaf/internal/kernel/kernel_storage_schema.h>
#include <orteaf/internal/kernel/mps/mps_kernel_args.h>
#include <orteaf/internal/kernel/mps/mps_kernel_base.h>
#include <orteaf/internal/kernel/mps/mps_kernel_entry.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session.h>
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
  kernel::Field<kernel::ParamId::NumElements, std::uint32_t> num_elements;

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
  // Extract storages and params
  auto storages = VectorAddStorages::extract(args);
  auto params = VectorAddParams::extract(args);

  // Begin session (auto cleanup on scope exit)
  auto session = mps_kernel::MpsKernelSession::begin(base, args, 0);
  if (!session)
    return;

  // Wait for input dependencies, bind, dispatch, update tokens
  session->waitDependencies(storages.a, storages.b, storages.c);
  session->bindStorages<0, 1, 2>(storages.a, storages.b, storages.c);
  session->bindParams<3>(params.num_elements);
  session->dispatch1D(params.num_elements);
  [[maybe_unused]] bool ok =
      session->updateTokens(storages.a, storages.b, storages.c);
  // RAII: auto endEncoding + commit
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
  entry.base.addKey("vector_add", "orteaf_vector_add");

  // Set the execute function
  entry.execute = vectorAddExecute;

  return entry;
}

} // namespace orteaf::extension::kernel::mps::ops

#endif // ORTEAF_ENABLE_MPS
