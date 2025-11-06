/**
 * @file cuda_check.h
 * @brief Utilities to validate CUDA Runtime/Driver API results.
 *
 * This header maps CUDA return values to `OrteafError` (thrown as
 * `std::system_error`) with detailed messages. Prefer using the macros.
 *
 * - Runtime API: `cuda_check`, `cuda_check_last`, `cuda_check_sync`
 * - Driver API:  `cu_driver_check`, `try_driver_call`
 * - Macros:      `CUDA_CHECK`, `CUDA_CHECK_LAST`, `CUDA_CHECK_SYNC`, `CU_CHECK`
 *
 * When CUDA is disabled, functions and macros are no-ops.
 */
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "orteaf/internal/diagnostics/error/error.h"

#if ORTEAF_ENABLE_CUDA
  #include <cuda_runtime_api.h>
  #include <cuda.h>
#endif

namespace orteaf::internal::backend::cuda {

#if ORTEAF_ENABLE_CUDA

/**
 * @brief Map a CUDA Runtime error to `OrteafErrc`.
 * @param err CUDA Runtime error code
 * @return Corresponding `OrteafErrc`
 */
inline orteaf::internal::diagnostics::error::OrteafErrc map_runtime_errc(cudaError_t err) {
    using orteaf::internal::diagnostics::error::OrteafErrc;
    switch (err) {
        case cudaErrorMemoryAllocation:
            return OrteafErrc::OutOfMemory;
        case cudaErrorInvalidValue:
            return OrteafErrc::InvalidArgument;
        case cudaErrorInitializationError:
        case cudaErrorNotInitialized:
            return OrteafErrc::BackendUnavailable;
        default:
            return OrteafErrc::OperationFailed;
    }
}

/**
 * @brief Check a CUDA Runtime call result.
 * @param err CUDA Runtime return value
 * @param expr Stringified expression (auto-filled by macro)
 * @param file Caller file (auto-filled by macro)
 * @param line Caller line (auto-filled by macro)
 * @throws std::system_error Mapped from `OrteafErrc` on failure.
 *
 * On error, includes CUDA error name/code, expression, and file/line.
 * Prefer using `CUDA_CHECK(expr)`.
 */
inline void cuda_check(cudaError_t err,
                       const char* expr,
                       const char* file,
                       int line) {
    if (err != cudaSuccess) {
        using namespace orteaf::internal::diagnostics::error;
        std::string msg = "CUDA error: ";
        msg += cudaGetErrorString(err);
        msg += " (code ";
        msg += std::to_string(static_cast<int>(err));
        msg += ") while calling ";
        msg += expr;
        msg += " at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        throw_error(map_runtime_errc(err), std::move(msg));
    }
}

/**
 * @brief Check `cudaGetLastError()`.
 * @param file Caller file (auto-filled by macro)
 * @param line Caller line (auto-filled by macro)
 * @throws std::system_error If a recent CUDA Runtime error exists.
 *
 * Useful right after kernel launches. Prefer `CUDA_CHECK_LAST()`.
 */
inline void cuda_check_last(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        using namespace orteaf::internal::diagnostics::error;
        std::string msg = "CUDA error: ";
        msg += cudaGetErrorString(err);
        msg += " (code ";
        msg += std::to_string(static_cast<int>(err));
        msg += ") while calling cudaGetLastError() at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        throw_error(map_runtime_errc(err), std::move(msg));
    }
}

/**
 * @brief Debug-only stream synchronization and validation.
 * @param stream CUDA stream to synchronize
 * @param file Caller file (auto-filled by macro)
 * @param line Caller line (auto-filled by macro)
 *
 * When `ORTEAF_DEBUG_CUDA_SYNC` is defined, calls
 * `cudaStreamSynchronize(stream)` and validates with `cuda_check`.
 * Otherwise this is a no-op.
 */
inline void cuda_check_sync(cudaStream_t stream,
                            const char* file,
                            int line) {
#ifdef ORTEAF_DEBUG_CUDA_SYNC
    (void)file; (void)line;
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stream)", file, line);
#else
    (void)stream; (void)file; (void)line;
#endif
}

/**
 * @brief Map a CUDA Driver error to `OrteafErrc`.
 * @param err CUDA Driver error code (`CUresult`)
 * @return Corresponding `OrteafErrc`
 */
inline orteaf::internal::diagnostics::error::OrteafErrc map_driver_errc(CUresult err) {
    using orteaf::internal::diagnostics::error::OrteafErrc;
    switch (err) {
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
            return OrteafErrc::BackendUnavailable;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return OrteafErrc::OutOfMemory;
        case CUDA_ERROR_INVALID_VALUE:
            return OrteafErrc::InvalidArgument;
        default:
            return OrteafErrc::OperationFailed;
    }
}

/**
 * @brief Check a CUDA Driver call result.
 * @param err Driver API return value
 * @param expr Stringified expression (auto-filled by macro)
 * @param file Caller file (auto-filled by macro)
 * @param line Caller line (auto-filled by macro)
 * @throws std::system_error Mapped from `OrteafErrc` on failure.
 *
 * When available, includes human-readable name/description via
 * `cuGetErrorName`/`cuGetErrorString`. Prefer `CU_CHECK(expr)`.
 */
inline void cu_driver_check(CUresult err,
                            const char* expr,
                            const char* file,
                            int line) {
    if (err != CUDA_SUCCESS) {
        using namespace orteaf::internal::diagnostics::error;
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(err, &name);
        cuGetErrorString(err, &msg);
        std::string result = "CUDA driver error: ";
        if (name) {
            result += name;
            result += " (";
        } else {
            result += "unknown (";
        }
        result += std::to_string(static_cast<int>(err));
        result += ")";
        if (msg) {
            result += ": ";
            result += msg;
        }
        result += " while calling ";
        result += expr;
        result += " at ";
        result += file;
        result += ":";
        result += std::to_string(line);
        throw_error(map_driver_errc(err), std::move(result));
    }
}

/**
 * @brief Try a Driver API call and absorb a recoverable error.
 * @tparam Fn Callable type (e.g., lambda)
 * @param fn Function to execute; may throw driver-related `std::system_error`
 * @return `true` on success; `false` if `CUDA_ERROR_DEINITIALIZED` is detected.
 * @throws std::system_error Re-thrown for non-recoverable driver errors.
 *
 * Returns `false` for `CUDA_ERROR_DEINITIALIZED` to allow re-initialization;
 * other errors are propagated.
 */
template <typename Fn>
bool try_driver_call(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
        return true;
    } catch (const std::system_error& ex) {
        std::string_view what = ex.what();
        if (what.find("CUDA_ERROR_DEINITIALIZED") != std::string_view::npos) {
            return false;
        }
        throw;
    }
}

#else  // !ORTEAF_ENABLE_CUDA

/**
 * @brief No-op stub when CUDA is disabled (runtime check).
 */
inline void cuda_check(int, const char*, const char*, int) noexcept {}
/**
 * @brief No-op stub when CUDA is disabled (last error check).
 */
inline void cuda_check_last(const char*, int) noexcept {}
/**
 * @brief No-op stub when CUDA is disabled (debug sync).
 */
inline void cuda_check_sync(void*, const char*, int) noexcept {}
/**
 * @brief No-op stub when CUDA is disabled (driver check).
 */
inline void cu_driver_check(int, const char*, const char*, int) noexcept {}

template <typename Fn>
bool try_driver_call(Fn&& fn) {
    std::forward<Fn>(fn)();
    return true;
}

#endif // ORTEAF_ENABLE_CUDA

} // namespace orteaf::internal::backend::cuda

#if ORTEAF_ENABLE_CUDA
  /**
   * @def CUDA_CHECK(expr)
   * @brief Validate a CUDA Runtime API result and throw on failure.
   */
  #define CUDA_CHECK(expr)       ::orteaf::internal::backend::cuda::cuda_check((expr), #expr, __FILE__, __LINE__)
  /**
   * @def CUDA_CHECK_LAST()
   * @brief Validate the most recent CUDA Runtime error state.
   */
  #define CUDA_CHECK_LAST()      ::orteaf::internal::backend::cuda::cuda_check_last(__FILE__, __LINE__)
  /**
   * @def CUDA_CHECK_SYNC(s)
   * @brief Synchronize a stream and validate only when `ORTEAF_DEBUG_CUDA_SYNC` is defined.
   */
  #define CUDA_CHECK_SYNC(s)     ::orteaf::internal::backend::cuda::cuda_check_sync((s), __FILE__, __LINE__)
  /**
   * @def CU_CHECK(expr)
   * @brief Validate a CUDA Driver API result and throw on failure.
   */
  #define CU_CHECK(expr)         ::orteaf::internal::backend::cuda::cu_driver_check((expr), #expr, __FILE__, __LINE__)
#else
  /** @def CUDA_CHECK(expr)  @brief No-op when CUDA is disabled. */
  #define CUDA_CHECK(expr)       (void)(expr)
  /** @def CUDA_CHECK_LAST() @brief No-op when CUDA is disabled. */
  #define CUDA_CHECK_LAST()      ((void)0)
  /** @def CUDA_CHECK_SYNC(s) @brief No-op when CUDA is disabled. */
  #define CUDA_CHECK_SYNC(s)     ((void)0)
  /** @def CU_CHECK(expr)     @brief No-op when CUDA is disabled. */
  #define CU_CHECK(expr)         ((void)0)
#endif
