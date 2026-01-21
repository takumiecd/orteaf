#pragma once

/**
 * @file storage_concepts.h
 * @brief Concepts for Storage type requirements.
 *
 * These concepts define the requirements for a Storage type to be
 * used with the TypedStorageManager template.
 */

#include <concepts>
#include <cstddef>

#include <orteaf/internal/execution/execution.h>

namespace orteaf::internal::storage::concepts {

/**
 * @brief Concept for Storage types that can be managed by TypedStorageManager.
 *
 * A Storage type must have:
 * - A BufferLease type (the underlying buffer lease)
 * - A Layout type (storage layout information)
 * - A static kExecution constant indicating the backend
 * - Methods: dtype(), numel(), sizeInBytes(), valid()
 */
template <typename T>
concept StorageConcept = requires {
  typename T::BufferLease;
  typename T::Layout;
  typename T::DType;
  {
    T::kExecution
  } -> std::convertible_to<::orteaf::internal::execution::Execution>;
} && requires(const T &storage) {
  { storage.dtype() } -> std::convertible_to<typename T::DType>;
  { storage.numel() } -> std::convertible_to<std::size_t>;
  { storage.sizeInBytes() } -> std::convertible_to<std::size_t>;
};

} // namespace orteaf::internal::storage::concepts
