/**
 * @file cuda_event.cu
 * @brief Implementation of CUDA event helpers (create/destroy/record/query/wait).
 */
#include "orteaf/internal/backend/cuda/cuda_event.h"
#include "orteaf/internal/backend/cuda/cuda_stats.h"
#include "orteaf/internal/backend/cuda/cuda_check.h"
#include "orteaf/internal/backend/cuda/cuda_objc_bridge.h"

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#include "orteaf/internal/diagnostics/error/error_impl.h"
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @copydoc orteaf::internal::backend::cuda::create_event
 */
CUevent_t create_event() {
#ifdef ORTEAF_ENABLE_CUDA
    CUevent event;
    CU_CHECK(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    update_create_event();
    return opaque_from_objc_noown<CUevent_t, CUevent>(event);
#else
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::destroy_event
 */
void destroy_event(CUevent_t event) {
#ifdef ORTEAF_ENABLE_CUDA
    if (event == nullptr) return;
    CUevent objc_event = objc_from_opaque_noown<CUevent>(event);
    CU_CHECK(cuEventDestroy(objc_event));
    update_destroy_event();
#else
    (void)event;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::record_event
 */
void record_event(CUevent_t event, CUstream_t stream) {
#ifdef ORTEAF_ENABLE_CUDA
    if (event == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "record_event: event cannot be nullptr");
    }
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "record_event: stream cannot be nullptr");
    }
    CUevent objc_event = objc_from_opaque_noown<CUevent>(event);
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CU_CHECK(cuEventRecord(objc_event, objc_stream));
#else
    (void)event;
    (void)stream;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::query_event
 */
bool query_event(CUevent_t event) {
#ifdef ORTEAF_ENABLE_CUDA
    if (event == nullptr) return true;
    CUevent objc_event = objc_from_opaque_noown<CUevent>(event);
    CUresult status = cuEventQuery(objc_event);
    if (status == CUDA_SUCCESS) return true;
    if (status == CUDA_ERROR_NOT_READY) return false;
    CU_CHECK(status);
    return false;
#else
    (void)event;
    return true;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::wait_event
 */
void wait_event(CUstream_t stream, CUevent_t event) {
#ifdef ORTEAF_ENABLE_CUDA
    if (stream == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "wait_event: stream cannot be nullptr");
    }
    if (event == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "wait_event: event cannot be nullptr");
    }
    CUstream objc_stream = objc_from_opaque_noown<CUstream>(stream);
    CUevent objc_event = objc_from_opaque_noown<CUevent>(event);
    CU_CHECK(cuStreamWaitEvent(objc_stream, objc_event, 0));
#else
    (void)stream;
    (void)event;
#endif
}

} // namespace orteaf::internal::backend::cuda
