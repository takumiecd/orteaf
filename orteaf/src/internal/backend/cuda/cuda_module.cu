/**
 * @file cuda_module.cu
 * @brief Implementation of CUDA module load/unload and function lookup.
 */
#include "orteaf/internal/backend/cuda/cuda_module.h"
#include "orteaf/internal/backend/cuda/cuda_objc_bridge.h"

#ifdef ORTEAF_ENABLE_CUDA
#include <cuda.h>
#include "orteaf/internal/backend/cuda/cuda_check.h"
#include "orteaf/internal/diagnostics/error/error_impl.h"
#endif

namespace orteaf::internal::backend::cuda {

/**
 * @copydoc orteaf::internal::backend::cuda::load_module_from_file
 */
CUmodule_t load_module_from_file(const char* filepath) {
#ifdef ORTEAF_ENABLE_CUDA
    if (filepath == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "load_module_from_file: filepath cannot be nullptr");
    }
    CUmodule module;
    CU_CHECK(cuModuleLoad(&module, filepath));
    return opaque_from_objc_noown<CUmodule_t, CUmodule>(module);
#else
    (void)filepath;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::load_module_from_image
 */
CUmodule_t load_module_from_image(const void* image) {
#ifdef ORTEAF_ENABLE_CUDA
    if (image == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "load_module_from_image: image cannot be nullptr");
    }
    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, image));
    return opaque_from_objc_noown<CUmodule_t, CUmodule>(module);
#else
    (void)image;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::get_function
 */
CUfunction_t get_function(CUmodule_t module, const char* kernel_name) {
#ifdef ORTEAF_ENABLE_CUDA
    if (module == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "get_function: module cannot be nullptr");
    }
    if (kernel_name == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throw_error(OrteafErrc::NullPointer, "get_function: kernel_name cannot be nullptr");
    }
    CUmodule objc_module = objc_from_opaque_noown<CUmodule>(module);
    CUfunction function;
    CU_CHECK(cuModuleGetFunction(&function, objc_module, kernel_name));
    return opaque_from_objc_noown<CUfunction_t, CUfunction>(function);
#else
    (void)module;
    (void)kernel_name;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::cuda::unload_module
 */
void unload_module(CUmodule_t module) {
#ifdef ORTEAF_ENABLE_CUDA
    if (module == nullptr) return;
    CUmodule objc_module = objc_from_opaque_noown<CUmodule>(module);
    CU_CHECK(cuModuleUnload(objc_module));
#else
    (void)module;
#endif
}

} // namespace orteaf::internal::backend::cuda
