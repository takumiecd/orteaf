#pragma once

#include "orteaf/internal/kernel/registry/kernel_registry.h"
#include "orteaf/internal/kernel/core/kernel_key.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace orteaf::internal::kernel::api {

/**
 * @brief Access the global KernelRegistry instance.
 */
::orteaf::internal::kernel::registry::KernelRegistry &kernelRegistry() noexcept;

/**
 * @brief Register a kernel with metadata.
 *
 * Convenience wrapper around kernelRegistry().registerKernel().
 *
 * @param key Kernel key for lookup
 * @param metadata Metadata for kernel reconstruction
 */
inline void registerKernel(KernelKey key, 
                          ::orteaf::internal::kernel::core::KernelMetadataLease metadata) {
  kernelRegistry().registerKernel(key, std::move(metadata));
}

/**
 * @brief Look up a kernel by key.
 *
 * Convenience wrapper around kernelRegistry().lookup().
 *
 * @param key Kernel key to look up
 * @return Pointer to entry, or nullptr if not found
 */
inline ::orteaf::internal::kernel::core::KernelEntry *lookupKernel(KernelKey key) {
  return kernelRegistry().lookup(key);
}

/**
 * @brief Check if a kernel is registered.
 *
 * Convenience wrapper around kernelRegistry().contains().
 *
 * @param key Kernel key to check
 * @return true if kernel is registered
 */
inline bool containsKernel(KernelKey key) {
  return kernelRegistry().contains(key);
}

} // namespace orteaf::internal::kernel::api
