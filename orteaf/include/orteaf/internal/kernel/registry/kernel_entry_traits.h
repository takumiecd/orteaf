#pragma once

#include <concepts>
#include <type_traits>

namespace orteaf::internal::kernel::registry {

/**
 * @brief Concept for kernel metadata types.
 *
 * Metadata must be able to rebuild an Entry from stored information.
 * This is used by Secondary Storage tier to reconstruct evicted entries.
 *
 * @tparam M Metadata type
 * @tparam E Entry type that metadata can rebuild
 */
template <typename M, typename E>
concept KernelMetadataConcept = requires(const M &m) {
  { m.rebuild() } -> std::convertible_to<E>;
};

/**
 * @brief Concept for kernel entry traits.
 *
 * Traits define the Entry and Metadata types for a specific execution backend.
 * This allows KernelRegistry to work with CPU, CUDA, MPS, or any future
 * backend.
 *
 * Requirements:
 * - Entry: The full kernel entry type (e.g., MpsKernelEntry)
 * - Metadata: Lightweight metadata for reconstruction (e.g., MpsKernelMetadata)
 * - toMetadata(): Static method to extract metadata from an entry
 *
 * @tparam T Traits type to check
 */
template <typename T>
concept KernelEntryTraitsConcept = requires {
  // Required type aliases
  typename T::Entry;
  typename T::Metadata;

  // Metadata must satisfy KernelMetadataConcept
  requires KernelMetadataConcept<typename T::Metadata, typename T::Entry>;

  // Static method to create metadata from entry
  {
    T::toMetadata(std::declval<const typename T::Entry &>())
  } -> std::same_as<typename T::Metadata>;
};

} // namespace orteaf::internal::kernel::registry
