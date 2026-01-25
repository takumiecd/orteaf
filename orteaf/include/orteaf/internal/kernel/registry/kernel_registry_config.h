#pragma once

#include <cstddef>

namespace orteaf::internal::kernel::registry {

/**
 * @brief Configuration for KernelRegistry capacity.
 *
 * Controls the capacity of each tier in the 3-tier cache hierarchy.
 */
struct KernelRegistryConfig {
  /// Maximum entries in Cache (L1) tier
  std::size_t cache_capacity{8};

  /// Maximum entries in Main Memory tier
  std::size_t main_memory_capacity{64};

  /// Secondary Storage is unbounded (no capacity limit)
};

} // namespace orteaf::internal::kernel::registry
