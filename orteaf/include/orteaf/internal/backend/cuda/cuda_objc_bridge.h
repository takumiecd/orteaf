/**
 * @file cuda_objc_bridge.h
 * @brief Opaque handle conversion helpers between CUDA Driver types and void*.
 *
 * Provides small inline utilities to convert between Objective-C/CUDA driver
 * pointer types and opaque handles used across the library. These helpers do
 * not transfer ownership and are simple `reinterpret_cast` wrappers.
 */
#pragma once

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace orteaf::internal::backend::cuda {

/**
 * @brief Convert a typed pointer to an opaque void* (no ownership transfer).
 * @tparam ObjcPtr Pointer type
 * @param p Pointer value
 * @return Opaque pointer as void*
 */
template <typename ObjcPtr>
static inline void* opaque_from_objc_noown(ObjcPtr p) noexcept {
    static_assert(std::is_pointer_v<ObjcPtr>, "ObjcPtr must be a pointer type");
    return reinterpret_cast<void*>(p);
}

/**
 * @brief Convert an opaque void* back to a typed pointer (no ownership).
 * @tparam ObjcPtr Pointer type
 * @param p Opaque pointer value
 * @return Reinterpreted typed pointer
 */
template <typename ObjcPtr>
static inline ObjcPtr objc_from_opaque_noown(void* p) noexcept {
    static_assert(std::is_pointer_v<ObjcPtr>, "ObjcPtr must be a pointer type");
    return reinterpret_cast<ObjcPtr>(p);
}

/**
 * @brief Convert a typed pointer to an opaque handle type.
 * @tparam Handle Opaque handle type (pointer-compatible)
 * @tparam ObjcPtr Pointer type
 * @param p Pointer value
 * @return Opaque handle value
 */
template <typename Handle, typename ObjcPtr>
static inline Handle opaque_from_objc_noown(ObjcPtr p) noexcept {
    return reinterpret_cast<Handle>(reinterpret_cast<void*>(p));
}

/**
 * @brief Convert an opaque handle back to a typed pointer.
 * @tparam Handle Opaque handle type (pointer-compatible)
 * @tparam ObjcPtr Pointer type
 * @param p Opaque handle value
 * @return Reinterpreted typed pointer
 */
template <typename Handle, typename ObjcPtr>
static inline ObjcPtr objc_from_opaque_noown(Handle p) noexcept {
    return reinterpret_cast<ObjcPtr>(reinterpret_cast<void*>(p));
}

/**
 * @brief Convert a `CUdeviceptr` to a 64-bit opaque integer.
 */
static inline std::uint64_t opaque_from_cu_deviceptr(CUdeviceptr p) noexcept {
    return static_cast<std::uint64_t>(p);
}

/**
 * @brief Convert a 64-bit opaque integer to `CUdeviceptr`.
 */
static inline CUdeviceptr cu_deviceptr_from_opaque(std::uint64_t p) noexcept {
    return static_cast<CUdeviceptr>(p);
}

/**
 * @brief Convert a `CUdevice` to an `int` opaque handle.
 */
static inline int opaque_from_cu_device(CUdevice p) noexcept {
    return static_cast<int>(p);
}

/**
 * @brief Convert an `int` opaque handle back to `CUdevice`.
 */
static inline CUdevice cu_device_from_opaque(int p) noexcept {
    return static_cast<CUdevice>(p);
}

} // namespace orteaf::internal::backend::cuda

#endif