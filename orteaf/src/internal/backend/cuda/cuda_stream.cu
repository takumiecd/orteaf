/**
 * @file cuda_stream.cu
 * @brief Implementation of CUDA stream helpers (create/destroy/sync/mem signals).
 */
#include "orteaf/internal/backend/cuda/cuda_stream.h"
#include "orteaf/internal/backend/cuda/cuda_check.h"
#include "orteaf/internal/backend/cuda/cuda_stats.h"
#include "orteaf/internal/backend/cuda/cuda_objc_bridge.h"

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @copydoc orteaf::internal::backend::cuda::get_stream
 */
CUstream_t get_stream() {
#ifdef ORTEAF_ENABLE_CUDA
    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    update_create_stream();
    return opaque_from_objc_noown<CUstream_t, CUstream>(stream);
#else
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::set_stream
 */
void set_stream(CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    (void)stream; // No driver API to set a global current stream.
#else
    (void)stream;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::release_stream
 */
void release_stream(CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    if (!stream) return;
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuStreamDestroy(objc_stream));
    update_destroy_stream();
#else
    (void)stream;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::synchronize_stream
 */
void synchronize_stream(CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuStreamSynchronize(objc_stream));
#else
    (void)stream;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::wait_stream
 */
void wait_stream(CUstream_t stream, CUdeviceptr_t addr, uint32_t value) {
#ifdef ORTEAF_ENABLE_CUDA
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CUdeviceptr objc_addr = cu_deviceptr_from_opaque(addr);
    CU_CHECK(cuStreamWaitValue32(objc_stream, objc_addr, value, CU_STREAM_WAIT_VALUE_GEQ));
#else
    (void)stream;
    (void)addr;
    (void)value;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::write_stream
 */
void write_stream(CUstream_t stream, CUdeviceptr_t addr, uint32_t value) {
#ifdef ORTEAF_ENABLE_CUDA
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CUdeviceptr objc_addr = cu_deviceptr_from_opaque(addr);
    CU_CHECK(cuStreamWriteValue32(objc_stream, objc_addr, value, CU_STREAM_WRITE_VALUE_DEFAULT));
#else
    (void)stream;
    (void)addr;
    (void)value;
#endif
}

} // namespace orteaf::internal::backend::cuda
