#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "orteaf/internal/diagnostics/error/error.h"

#ifdef ORTEAF_ENABLE_CUDA
  #include <cuda_runtime_api.h>  // 最小限でOK（<cuda_runtime.h>でも可）
  #include <cuda.h>
#endif

namespace orteaf::internal::backend::cuda {

#ifdef ORTEAF_ENABLE_CUDA

// 内部: CUDA ランタイムエラーのマッピング
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

// 戻り値チェック（失敗なら OrteafError を送出）
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

// 直近のエラー確認（カーネル起動後などに便利）
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

// デバッグ用：ストリーム同期してエラー顕在化（非同期APIのときに有用）
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

// 内部: CUDA ドライバエラーのマッピング
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

template <typename Fn>
bool try_driver_call(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
        return true;
    } catch (const std::system_error& ex) {
        // CUDA_ERROR_DEINITIALIZED の場合のみ false を返す（再初期化を許可）
        std::string_view what = ex.what();
        if (what.find("CUDA_ERROR_DEINITIALIZED") != std::string_view::npos) {
            return false;
        }
        throw;
    }
}

#else  // !ORTEAF_ENABLE_CUDA

// 非 CUDA ビルド：何もしないダミー定義（型にも触れない）
inline void cuda_check(int, const char*, const char*, int) noexcept {}
inline void cuda_check_last(const char*, int) noexcept {}
inline void cuda_check_sync(void*, const char*, int) noexcept {}
inline void cu_driver_check(int, const char*, const char*, int) noexcept {}

template <typename Fn>
bool try_driver_call(Fn&& fn) {
    std::forward<Fn>(fn)();
    return true;
}

#endif // ORTEAF_ENABLE_CUDA

} // namespace orteaf::internal::backend::cuda

// 使いやすいマクロ
#ifdef ORTEAF_ENABLE_CUDA
  #define CUDA_CHECK(expr)       ::orteaf::internal::backend::cuda::cuda_check((expr), #expr, __FILE__, __LINE__)
  #define CUDA_CHECK_LAST()      ::orteaf::internal::backend::cuda::cuda_check_last(__FILE__, __LINE__)
  #define CUDA_CHECK_SYNC(s)     ::orteaf::internal::backend::cuda::cuda_check_sync((s), __FILE__, __LINE__)
  #define CU_CHECK(expr)         ::orteaf::internal::backend::cuda::cu_driver_check((expr), #expr, __FILE__, __LINE__)
#else
  #define CUDA_CHECK(expr)       (void)(expr)
  #define CUDA_CHECK_LAST()      ((void)0)
  #define CUDA_CHECK_SYNC(s)     ((void)0)
  #define CU_CHECK(expr)         ((void)0)
#endif
