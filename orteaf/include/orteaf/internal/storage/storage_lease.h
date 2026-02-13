#pragma once

/**
 * @file storage_lease.h
 * @brief Type-erased Storage Lease.
 *
 * Provides a unified interface for working with storage leases
 * from different backends (CPU, MPS, etc.).
 */

#include <cstddef>
#include <utility>
#include <variant>

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/dtype/dtype.h>
#include <orteaf/internal/execution/execution.h>
#include <orteaf/internal/storage/registry/storage_types.h>

namespace orteaf::internal::storage {

/**
 * @brief Type-erased storage lease container.
 *
 * Wraps backend-specific StorageLease types in a std::variant,
 * providing a unified interface similar to the KernelArgs pattern.
 */
class StorageLease {
public:
  using DType = ::orteaf::internal::DType;
  using Execution = ::orteaf::internal::execution::Execution;
  using CpuLease = registry::CpuStorageLease;
#if ORTEAF_ENABLE_MPS
  using MpsLease = registry::MpsStorageLease;
#endif
#if ORTEAF_ENABLE_CUDA
  using CudaLease = registry::CudaStorageLease;
#endif

  // Variant type holding all backend lease implementations
  using Variant = std::variant<std::monostate, CpuLease
#if ORTEAF_ENABLE_MPS
                               ,
                               MpsLease
#endif
#if ORTEAF_ENABLE_CUDA
                               ,
                               CudaLease
#endif
                               >;

  /**
   * @brief Default constructor. Creates an invalid (uninitialized)
   * StorageLease.
   */
  StorageLease() = default;

  /**
   * @brief Type-erase a backend-specific StorageLease.
   *
   * @tparam T A backend-specific StorageLease type.
   * @param lease The backend-specific lease to wrap.
   * @return A new StorageLease instance containing the wrapped lease.
   */
  template <typename T> static StorageLease erase(T lease) {
    return StorageLease(Variant{std::move(lease)});
  }

  /**
   * @brief Attempt to retrieve as a specific type.
   *
   * @tparam T The target lease type.
   * @return Pointer if it holds type T, nullptr otherwise.
   */
  template <typename T> T *tryAs() { return std::get_if<T>(&variant_); }

  template <typename T> const T *tryAs() const {
    return std::get_if<T>(&variant_);
  }

  /**
   * @brief Apply a visitor to the underlying lease.
   */
  template <typename Visitor> decltype(auto) visit(Visitor &&v) {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  template <typename Visitor> decltype(auto) visit(Visitor &&v) const {
    return std::visit(std::forward<Visitor>(v), variant_);
  }

  /**
   * @brief Check if the StorageLease is valid.
   */
  bool valid() const {
    return !std::holds_alternative<std::monostate>(variant_);
  }

  /**
   * @brief Return the execution backend for this StorageLease.
   */
  Execution execution() const {
    return std::visit(
        [](const auto &lease) -> Execution {
          using T = std::decay_t<decltype(lease)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get execution from uninitialized storage lease");
            return Execution::Cpu; // Unreachable, but needed for compilation
          } else {
            if (!lease.operator->()) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Cannot get execution from invalid storage lease");
              return Execution::Cpu; // Unreachable, but needed for compilation
            }
            return lease->execution();
          }
        },
        variant_);
  }

  /**
   * @brief Return the data type of elements in this storage lease.
   */
  DType dtype() const {
    return std::visit(
        [](const auto &lease) -> DType {
          using T = std::decay_t<decltype(lease)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get dtype from uninitialized storage lease");
            return DType::F32; // Unreachable, but needed for compilation
          } else {
            if (!lease.operator->()) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Cannot get dtype from invalid storage lease");
              return DType::F32; // Unreachable, but needed for compilation
            }
            return lease->dtype();
          }
        },
        variant_);
  }

  /**
   * @brief Return the number of elements in this storage lease.
   */
  std::size_t numel() const {
    return std::visit(
        [](const auto &lease) -> std::size_t {
          using T = std::decay_t<decltype(lease)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get numel from uninitialized storage lease");
            return 0; // Unreachable, but needed for compilation
          } else {
            if (!lease.operator->()) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Cannot get numel from invalid storage lease");
              return 0; // Unreachable, but needed for compilation
            }
            return lease->numel();
          }
        },
        variant_);
  }

  /**
   * @brief Return the size of the storage in bytes.
   */
  std::size_t sizeInBytes() const {
    return std::visit(
        [](const auto &lease) -> std::size_t {
          using T = std::decay_t<decltype(lease)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            ::orteaf::internal::diagnostics::error::throwError(
                ::orteaf::internal::diagnostics::error::OrteafErrc::
                    InvalidState,
                "Cannot get size from uninitialized storage lease");
            return 0; // Unreachable, but needed for compilation
          } else {
            if (!lease.operator->()) {
              ::orteaf::internal::diagnostics::error::throwError(
                  ::orteaf::internal::diagnostics::error::OrteafErrc::
                      InvalidState,
                  "Cannot get size from invalid storage lease");
              return 0; // Unreachable, but needed for compilation
            }
            return lease->sizeInBytes();
          }
        },
        variant_);
  }

  explicit operator bool() const noexcept { return valid(); }

private:
  explicit StorageLease(Variant v) : variant_(std::move(v)) {}

  Variant variant_{};
};

} // namespace orteaf::internal::storage
