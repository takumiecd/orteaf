#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <cstdint>

#include <orteaf/internal/kernel/schema/kernel_param_schema.h>
#include <orteaf/internal/kernel/schema/kernel_storage_schema.h>
#include <orteaf/internal/kernel/core/kernel_args.h>
#include <orteaf/internal/kernel/core/kernel_entry.h>
#include <orteaf/internal/kernel/mps/mps_kernel_session.h>
#include <orteaf/internal/kernel/param/param_id.h>
#include <orteaf/internal/kernel/storage/operand_id.h>

namespace orteaf::extension::kernel::mps::ops {

namespace kernel = ::orteaf::internal::kernel;
namespace mps_kernel = ::orteaf::internal::kernel::mps;
namespace mps_resource = ::orteaf::internal::execution::mps::resource;

/**
 * @brief Storage schema for vector add kernel.
 *
 * Defines the required buffer bindings:
 * - Input0: First input vector (A)
 * - Input1: Second input vector (B)
 * - Output: Output vector (C = A + B)
 */
struct VectorAddStorages : kernel::StorageSchema<VectorAddStorages> {
  kernel::StorageField<kernel::OperandId::Input0> a;
  kernel::StorageField<kernel::OperandId::Input1> b;
  kernel::StorageField<kernel::OperandId::Output> c;

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
 * This function is called by KernelEntry::run().
 *
 * @param lease KernelBaseLease reference (may hold an MpsKernelBase lease)
 * @param args Kernel arguments containing storages and parameters
 */
inline mps_resource::MpsKernelBase *
getMpsBase(kernel::core::KernelEntry::KernelBaseLease &lease) {
  using LeaseT = kernel::core::KernelEntry::MpsKernelBaseLease;
  auto *mps_lease = std::get_if<LeaseT>(&lease);
  if (!mps_lease || !(*mps_lease)) {
    return nullptr;
  }
  return mps_lease->operator->();
}

inline void vectorAddExecute(kernel::core::KernelEntry::KernelBaseLease &lease,
                             ::orteaf::internal::kernel::KernelArgs &args) {
  auto *base_ptr = getMpsBase(lease);
  if (!base_ptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS kernel base lease is invalid");
  }
  // Extract storages and params
  auto storages = VectorAddStorages::extract(args);
  auto params = VectorAddParams::extract(args);

  // Begin session (auto cleanup on scope exit)
  auto session = mps_kernel::MpsKernelSession::begin(*base_ptr, args, 0);
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
 * This factory function creates a KernelEntry configured for vector add.
 *
 * @param lease Kernel base lease with registered library/function keys
 * @return KernelEntry for vector add operations
 */
inline kernel::core::KernelEntry createVectorAddKernel(
    kernel::core::KernelEntry::MpsKernelBaseLease lease) {
  kernel::core::KernelEntry entry;
  entry.setBase(
      kernel::core::KernelEntry::KernelBaseLease{std::move(lease)});
  entry.setExecute(vectorAddExecute);

  return entry;
}

} // namespace orteaf::extension::kernel::mps::ops

#endif // ORTEAF_ENABLE_MPS
