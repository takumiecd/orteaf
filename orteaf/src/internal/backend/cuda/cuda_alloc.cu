#include "orteaf/internal/backend/cuda/cuda_alloc.h"
#include "orteaf/internal/backend/cuda/cuda_check.h"
#include "orteaf/internal/backend/cuda/cuda_stats.h"
#include "orteaf/internal/backend/cuda/cuda_objc_bridge.h"

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#include "orteaf/internal/diagnostics/error/error_impl.h"
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @brief Allocate device memory on CUDA device.
 *
 * Implementation of alloc() declared in cuda_alloc.h.
 * Uses cuMemAlloc to allocate device memory and updates statistics.
 *
 * @param size Size of memory to allocate in bytes.
 * @return Opaque CUDA device pointer. Returns 0 if CUDA is not available.
 * @throws std::runtime_error If CUDA allocation fails (when ORTEAF_ENABLE_CUDA is defined).
 */
CUdeviceptr_t alloc(size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    if (size == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::InvalidParameter, "alloc: size cannot be 0");
    }
    CUdeviceptr ptr;
    CU_CHECK(cuMemAlloc(&ptr, size));
    update_alloc(size);
    return opaque_from_cu_deviceptr(ptr);
#else
    (void)size;
    return 0;
#endif
}

/**
 * @brief Free device memory on CUDA device.
 *
 * Implementation of free() declared in cuda_alloc.h.
 * Uses cuMemFree to free device memory and updates statistics.
 *
 * @param ptr Opaque CUDA device pointer to free.
 * @param size Size of memory to free in bytes. Used for statistics update.
 */
void free(CUdeviceptr_t ptr, size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    CUdeviceptr objc_ptr = cu_deviceptr_from_opaque(ptr);
    CU_CHECK(cuMemFree(objc_ptr));
    update_dealloc(size);
#else
    (void)ptr;
#endif
}

/**
 * @brief Allocate device memory on CUDA device asynchronously.
 *
 * Implementation of alloc_stream() declared in cuda_alloc.h.
 * Uses cuMemAllocAsync to allocate device memory asynchronously on the specified stream.
 *
 * @param size Size of memory to allocate in bytes.
 * @param stream CUDA stream handle for asynchronous allocation.
 * @return Opaque CUDA device pointer. Returns 0 if CUDA is not available.
 * @throws std::runtime_error If stream is nullptr or CUDA allocation fails.
 */
CUdeviceptr_t alloc_stream(size_t size, CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    if (size == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::InvalidParameter, "alloc_stream: size cannot be 0");
    }
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "alloc_stream: stream cannot be nullptr");
    }
    CUdeviceptr ptr;
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuMemAllocAsync(&ptr, size, objc_stream));
    update_alloc(size);
    return opaque_from_cu_deviceptr(ptr);
#else
    (void)size;
    (void)stream;
    return 0;
#endif
}

/**
 * @brief Free device memory on CUDA device asynchronously.
 *
 * Implementation of free_stream() declared in cuda_alloc.h.
 * Uses cuMemFreeAsync to free device memory asynchronously on the specified stream.
 *
 * @param ptr Opaque CUDA device pointer to free.
 * @param size Size of memory to free in bytes. Used for statistics update.
 * @param stream CUDA stream handle for asynchronous deallocation.
 * @throws std::runtime_error If stream is nullptr or CUDA deallocation fails.
 */
void free_stream(CUdeviceptr_t ptr, size_t size, CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    CUdeviceptr objc_ptr = cu_deviceptr_from_opaque(ptr);
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "free_stream: stream cannot be nullptr");
    }
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuMemFreeAsync(objc_ptr, objc_stream));
    update_dealloc(size);
#else
    (void)ptr;
    (void)stream;
#endif
}

/**
 * @brief Allocate pinned host memory.
 *
 * Implementation of alloc_host() declared in cuda_alloc.h.
 * Uses cuMemAllocHost to allocate page-locked (pinned) host memory.
 *
 * @param size Size of memory to allocate in bytes.
 * @return Pointer to allocated pinned host memory. Returns nullptr if CUDA is not available.
 * @throws std::runtime_error If CUDA allocation fails (when ORTEAF_ENABLE_CUDA is defined).
 */
void* alloc_host(std::size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    if (size == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::InvalidParameter, "alloc_host: size cannot be 0");
    }
    void* ptr;
    CU_CHECK(cuMemAllocHost(&ptr, size));
    update_alloc(size);
    return ptr;
#else
    (void)size;
    return nullptr;
#endif
}

/**
 * @brief Copy data from device to host memory.
 *
 * Implementation of copy_to_host() declared in cuda_alloc.h.
 * Uses cuMemcpyDtoH to copy data synchronously from device to host memory.
 * This operation does not allocate memory and does not update statistics.
 *
 * @param ptr Opaque CUDA device pointer to copy from.
 * @param host_ptr Host memory pointer to copy to.
 * @param size Number of bytes to copy.
 * @throws std::runtime_error If CUDA copy operation fails (when ORTEAF_ENABLE_CUDA is defined).
 */
void copy_to_host(CUdeviceptr_t ptr, void* host_ptr, size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    if (ptr == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::InvalidParameter, "copy_to_host: ptr cannot be 0");
    }
    if (host_ptr == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "copy_to_host: host_ptr cannot be nullptr");
    }
    CUdeviceptr objc_ptr = cu_deviceptr_from_opaque(ptr);
    CU_CHECK(cuMemcpyDtoH(host_ptr, objc_ptr, size));
#else
    (void)ptr;
    (void)host_ptr;
    (void)size;
#endif
}

/**
 * @brief Copy data from host to device memory.
 *
 * Implementation of copy_to_device() declared in cuda_alloc.h.
 * Uses cuMemcpyHtoD to copy data synchronously from host to device memory.
 * This operation does not allocate memory and does not update statistics.
 *
 * @param host_ptr Host memory pointer to copy from.
 * @param ptr Opaque CUDA device pointer to copy to.
 * @param size Number of bytes to copy.
 * @throws std::runtime_error If CUDA copy operation fails (when ORTEAF_ENABLE_CUDA is defined).
 */
void copy_to_device(void* host_ptr, CUdeviceptr_t ptr, size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    if (ptr == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::InvalidParameter, "copy_to_device: ptr cannot be 0");
    }
    if (host_ptr == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "copy_to_device: host_ptr cannot be nullptr");
    }
    CUdeviceptr objc_ptr = cu_deviceptr_from_opaque(ptr);
    CU_CHECK(cuMemcpyHtoD(objc_ptr, host_ptr, size));
#else
    (void)host_ptr;
    (void)ptr;
    (void)size;
#endif
}

/**
 * @brief Free pinned host memory.
 *
 * Implementation of free_host() declared in cuda_alloc.h.
 * Uses cuMemFreeHost to free page-locked (pinned) host memory.
 *
 * @param ptr Pointer to pinned host memory to free.
 * @param size Size of memory to free in bytes. Used for statistics update.
 * @throws std::runtime_error If CUDA deallocation fails (when ORTEAF_ENABLE_CUDA is defined).
 */
void free_host(void* ptr, size_t size) {
#ifdef ORTEAF_ENABLE_CUDA
    if (ptr == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "free_host: ptr cannot be nullptr");
    }
    CU_CHECK(cuMemFreeHost(ptr));
    update_dealloc(size);
#else
    (void)ptr;
#endif
}

} // namespace orteaf::internal::backend::cuda
