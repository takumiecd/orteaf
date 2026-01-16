#pragma once

/**
 * @file storage.h
 * @brief Type-erased storage container for backend-agnostic tensor data.
 *
 * The Storage class provides a unified interface for managing tensor storage
 * across different execution backends (CPU, MPS, CUDA). It uses type erasure
 * via std::variant to hold backend-specific storage objects while exposing
 * a common API.
 *
 * @par Key Features
 * - Type erasure: Wrap any backend-specific storage into a uniform Storage type
 * - Type recovery: Retrieve the underlying storage via tryAs<T>()
 * - Visitor pattern: Apply operations via visit() without knowing the concrete
 * type
 * - Validity checking: Check if storage is initialized via valid()
 *
 * @par Example Usage
 * @code
 * // Create from backend-specific storage
 * CpuStorage cpu_storage{...};
 * Storage storage = Storage::erase(std::move(cpu_storage));
 *
 * // Check validity
 * if (storage.valid()) {
 *     // Try to get as specific type
 *     if (auto* cpu = storage.tryAs<CpuStorage>()) {
 *         // Use cpu storage...
 *     }
 *
 *     // Or use visitor pattern
 *     storage.visit([](auto& s) {
 *         using T = std::decay_t<decltype(s)>;
 *         if constexpr (std::is_same_v<T, CpuStorage>) {
 *             // Handle CPU storage
 *         }
 *     });
 * }
 * @endcode
 *
 * @see StorageType
 * @see StorageVariant
 */

#include <cstddef>
#include <utility>

#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/storage_types.h>

namespace orteaf::internal::storage {

/**
 * @brief Type-erased container for backend-specific storage objects.
 *
 * Storage wraps a StorageVariant internally, providing a clean API for
 * working with tensor storage without exposing the variant details.
 * A default-constructed Storage is in an invalid (uninitialized) state,
 * represented internally by std::monostate.
 */
class Storage {
public:
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;

  /**
   * @brief Default constructor. Creates an invalid (uninitialized) storage.
   *
   * After default construction, valid() returns false.
   */
  Storage() = default;

  /**
   * @brief Type-erase a backend-specific storage object.
   *
   * Creates a Storage instance from a concrete storage type. The storage
   * is moved into the internal variant.
   *
   * @tparam T A type satisfying the StorageType concept.
   * @param storage The backend-specific storage to wrap.
   * @return A new Storage instance containing the wrapped storage.
   *
   * @par Example
   * @code
   * CpuStorage cpu_storage{...};
   * Storage s = Storage::erase(std::move(cpu_storage));
   * assert(s.valid());
   * @endcode
   */
  template <StorageType T> static Storage erase(T storage) {
    return Storage(StorageVariant{std::move(storage)});
  }

  /**
   * @brief Attempt to retrieve the storage as a specific type.
   *
   * @tparam T The target storage type (must satisfy StorageType).
   * @return Pointer to the storage if it holds type T, nullptr otherwise.
   *
   * @par Example
   * @code
   * if (auto* cpu = storage.tryAs<CpuStorage>()) {
   *     // storage contains a CpuStorage
   * }
   * @endcode
   */
  template <StorageType T> std::remove_cvref_t<T> *tryAs() {
    return std::get_if<std::remove_cvref_t<T>>(&storage_);
  }

  /**
   * @brief Attempt to retrieve the storage as a specific type (const version).
   *
   * @tparam T The target storage type (must satisfy StorageType).
   * @return Const pointer to the storage if it holds type T, nullptr otherwise.
   */
  template <StorageType T> const std::remove_cvref_t<T> *tryAs() const {
    return std::get_if<std::remove_cvref_t<T>>(&storage_);
  }

  /**
   * @brief Check if the storage is initialized.
   *
   * A storage is valid if it contains an actual backend storage,
   * not std::monostate (the uninitialized state).
   *
   * @return true if storage contains a valid backend storage, false otherwise.
   */
  bool valid() const {
    return !std::holds_alternative<std::monostate>(storage_);
  }

  /**
   * @brief Return the execution backend for this storage.
   *
   * @return The Execution enum value indicating the backend.
   * @throws If the storage is uninitialized (monostate).
   */
  Execution execution() const {
    return std::visit(
        [](const auto &s) -> Execution {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get execution from uninitialized storage");
            return Execution::Cpu; // Unreachable, but needed for compilation
          } else {
            return T::kExecution;
          }
        },
        storage_);
  }

  /**
   * @brief Return the data type of elements in this storage.
   *
   * @return The DType enum value.
   * @throws If the storage is uninitialized (monostate).
   */
  DType dtype() const {
    return std::visit(
        [](const auto &s) -> DType {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get dtype from uninitialized storage");
            return DType::F32; // Unreachable, but needed for compilation
          } else {
            return s.dtype();
          }
        },
        storage_);
  }

  /**
   * @brief Return the size of the storage in bytes.
   *
   * @return Size in bytes.
   * @throws If the storage is uninitialized (monostate).
   */
  std::size_t sizeInBytes() const {
    return std::visit(
        [](const auto &s) -> std::size_t {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get size from uninitialized storage");
            return 0; // Unreachable, but needed for compilation
          } else {
            return s.sizeInBytes();
          }
        },
        storage_);
  }

  /**
   * @brief Apply a visitor to the underlying storage.
   *
   * Uses std::visit to apply the visitor to the contained storage type.
   * The visitor must handle all possible types in StorageVariant,
   * including std::monostate.
   *
   * @tparam Visitor A callable type that can handle all StorageVariant
   * alternatives.
   * @param visitor The visitor to apply.
   * @return The result of invoking the visitor on the contained storage.
   *
   * @par Example
   * @code
   * storage.visit([](auto& s) {
   *     using T = std::decay_t<decltype(s)>;
   *     if constexpr (std::is_same_v<T, std::monostate>) {
   *         // Handle uninitialized
   *     } else if constexpr (std::is_same_v<T, CpuStorage>) {
   *         // Handle CPU
   *     }
   * });
   * @endcode
   */
  template <class Visitor> decltype(auto) visit(Visitor &&visitor) {
    return std::visit(std::forward<Visitor>(visitor), storage_);
  }

  /**
   * @brief Apply a visitor to the underlying storage (const version).
   *
   * @tparam Visitor A callable type that can handle all StorageVariant
   * alternatives.
   * @param visitor The visitor to apply.
   * @return The result of invoking the visitor on the contained storage.
   */
  template <class Visitor> decltype(auto) visit(Visitor &&visitor) const {
    return std::visit(std::forward<Visitor>(visitor), storage_);
  }

private:
  /**
   * @brief Private constructor from a StorageVariant.
   * @param storage The variant to wrap.
   */
  explicit Storage(StorageVariant storage) : storage_(std::move(storage)) {}

  /** @brief The underlying variant holding the backend-specific storage. */
  StorageVariant storage_{}; // Default: monostate (uninitialized)
};

} // namespace orteaf::internal::storage
