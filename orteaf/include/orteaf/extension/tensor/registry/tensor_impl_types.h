#pragma once

/**
 * @file tensor_impl_types.h
 * @brief Registration of TensorImpl types.
 *
 * This file is where contributors register new TensorImpl types.
 * Managers are auto-generated via TensorImplManager<Impl>.
 *
 * Adding to RegisteredImpls is all you need - everything else is automatic.
 */

#include <orteaf/extension/tensor/dense_tensor_impl.h>
#include <orteaf/internal/tensor/registry/tensor_impl_registry.h>

namespace orteaf::internal::tensor::registry {

// =============================================================================
// Registered TensorImpl Types
// =============================================================================

using RegisteredImpls =
    TensorImplRegistry<::orteaf::extension::tensor::DenseTensorImpl
                       // Contributors: Add new impls here
                       >;

} // namespace orteaf::internal::tensor::registry

// Re-export for convenience
namespace orteaf::extension::tensor::registry {
using RegisteredImpls = ::orteaf::internal::tensor::registry::RegisteredImpls;
} // namespace orteaf::extension::tensor::registry
