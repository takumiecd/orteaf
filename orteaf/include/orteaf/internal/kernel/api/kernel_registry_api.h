#pragma once

#include "orteaf/internal/kernel/registry/kernel_registry.h"
#include "orteaf/internal/kernel/core/kernel_key.h"
#include "orteaf/internal/kernel/core/kernel_metadata.h"

namespace orteaf::internal::kernel::api {

/**
 * @brief API for kernel registry management.
 *
 * Provides access to the global KernelRegistry instance and convenience
 * methods for kernel registration and lookup.
 *
 * This is a static-only class following the same pattern as TensorApi,
 * CpuExecutionApi, etc.
 */
class KernelRegistryApi {
public:
  using Registry = ::orteaf::internal::kernel::registry::KernelRegistry;
  using Entry = ::orteaf::internal::kernel::core::KernelEntry;
  using MetadataLease = ::orteaf::internal::kernel::core::KernelMetadataLease;

  KernelRegistryApi() = delete;

  /**
   * @brief Access the global KernelRegistry instance.
   */
  static Registry &instance() noexcept;

  /**
   * @brief Register a kernel with metadata.
   *
   * @param key Kernel key for lookup
   * @param metadata Metadata for kernel reconstruction
   */
  static void registerKernel(KernelKey key, MetadataLease metadata) {
    instance().registerKernel(key, std::move(metadata));
  }

  /**
   * @brief Look up a kernel by key.
   *
   * @param key Kernel key to look up
   * @return Pointer to entry, or nullptr if not found
   */
  static Entry *lookupKernel(KernelKey key) {
    return instance().lookup(key);
  }

  /**
   * @brief Check if a kernel is registered.
   *
   * @param key Kernel key to check
   * @return true if kernel is registered
   */
  static bool containsKernel(KernelKey key) {
    return instance().contains(key);
  }

  /**
   * @brief Prefetch a kernel into cache.
   *
   * @param key Kernel key to prefetch
   * @return true if kernel was found and prefetched
   */
  static bool prefetchKernel(KernelKey key) {
    return instance().prefetch(key);
  }

  /**
   * @brief Flush the cache tier to main memory.
   */
  static void flushCache() {
    instance().flush();
  }

  /**
   * @brief Clear all tiers of the registry.
   */
  static void clear() {
    instance().clear();
  }

  /**
   * @brief Get registry statistics.
   */
  static const Registry::Stats &stats() {
    return instance().stats();
  }
};

} // namespace orteaf::internal::kernel::api
