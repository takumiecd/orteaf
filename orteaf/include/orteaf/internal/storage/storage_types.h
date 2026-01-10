#pragma once

/**
 * @file storage_types.h
 * @brief Type definitions for the type-erased Storage system.
 *
 * This header provides the central type definitions used by the Storage class:
 * - StorageVariant: A std::variant that can hold any backend-specific storage
 * type
 * - StorageType: A concept that constrains template parameters to valid storage
 * types
 *
 * All conditional compilation for different backends (CPU, MPS, CUDA) is
 * centralized in this file, keeping the main Storage class clean and readable.
 *
 * @see Storage
 */

#include <type_traits>
#include <variant>

#include <orteaf/internal/storage/cpu/cpu_storage.h>
#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/storage/mps/mps_storage.h>
#endif // ORTEAF_ENABLE_MPS

namespace orteaf::internal::storage {

/**
 * @name Type Aliases
 * @brief Backend-specific storage type aliases for convenience.
 * @{
 */

/** @brief CPU backend storage type. Always available. */
using CpuStorage = ::orteaf::internal::storage::cpu::CpuStorage;

#if ORTEAF_ENABLE_MPS
/** @brief MPS (Metal Performance Shaders) backend storage type. Only available
 * on Apple platforms. */
using MpsStorage = ::orteaf::internal::storage::mps::MpsStorage;
#endif // ORTEAF_ENABLE_MPS

/** @} */

/**
 * @brief Variant type that can hold any backend-specific storage or be
 * uninitialized.
 *
 * The StorageVariant is the underlying type used by the Storage class for type
 * erasure. It includes:
 * - std::monostate: Represents an uninitialized/empty storage state
 * - CpuStorage: CPU backend storage (always available)
 * - MpsStorage: MPS backend storage (when ORTEAF_ENABLE_MPS is defined)
 *
 * Future backends (e.g., CUDA) should be added here.
 *
 * @note std::monostate is the first alternative, so default-constructed
 * StorageVariant represents an uninitialized state.
 */
using StorageVariant = std::variant<std::monostate, // Uninitialized state
                                    CpuStorage
#if ORTEAF_ENABLE_MPS
                                    ,
                                    MpsStorage
#endif // ORTEAF_ENABLE_MPS
                                    >;

namespace detail {

/**
 * @brief Type trait to check if a type T is a member of a std::variant.
 * @tparam T The type to check.
 * @tparam Variant The variant type to check against.
 */
template <class T, class Variant> struct is_variant_member : std::false_type {};

/**
 * @brief Specialization for std::variant.
 * @tparam T The type to check.
 * @tparam Ts The types in the variant.
 */
template <class T, class... Ts>
struct is_variant_member<T, std::variant<Ts...>>
    : std::disjunction<std::is_same<std::remove_cvref_t<T>, Ts>...> {};

} // namespace detail

/**
 * @brief Concept that constrains types to valid storage types.
 *
 * A type satisfies StorageType if:
 * 1. It is a member of StorageVariant (after removing cv-qualifiers and
 * references)
 * 2. It is NOT std::monostate (which represents the uninitialized state)
 *
 * @tparam T The type to check.
 *
 * @par Example
 * @code
 * static_assert(StorageType<CpuStorage>);           // OK
 * static_assert(StorageType<CpuStorage&>);          // OK (reference is
 * stripped) static_assert(StorageType<const CpuStorage&>);    // OK
 * static_assert(!StorageType<std::monostate>);      // Fails (monostate
 * excluded) static_assert(!StorageType<int>);                 // Fails (not in
 * variant)
 * @endcode
 */
template <class T>
concept StorageType =
    detail::is_variant_member<std::remove_cvref_t<T>, StorageVariant>::value &&
    !std::is_same_v<std::remove_cvref_t<T>, std::monostate>;

} // namespace orteaf::internal::storage
