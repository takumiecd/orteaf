/**
 * @file cuda_dim.cu
 * @brief Implementation for CUDA dimension helper PODs and converters.
 */
#include "orteaf/internal/backend/cuda/cuda_dim.h"

namespace orteaf::internal::backend::cuda {

/**
 * @copydoc orteaf::internal::backend::cuda::make_dim3
 */
CudaDim3_t make_dim3(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return CudaDim3_t{x, y, z};
}

/**
 * @copydoc orteaf::internal::backend::cuda::make_uint3
 */
CudaUInt3_t make_uint3(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return CudaUInt3_t{x, y, z};
}

#ifndef ORTEAF_ENABLE_CUDA
/** Convert `CudaDim3_t` to CUDA `dim3`. */
dim3 to_cuda_dim3(CudaDim3_t value) noexcept {
    dim3 result{};
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

/** Convert CUDA `dim3` to `CudaDim3_t`. */
CudaDim3_t from_cuda_dim3(dim3 value) noexcept {
    return CudaDim3_t{value.x, value.y, value.z};
}

/** Convert `CudaUInt3_t` to CUDA `uint3`. */
uint3 to_cuda_uint3(CudaUInt3_t value) noexcept {
    uint3 result{};
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

/** Convert CUDA `uint3` to `CudaUInt3_t`. */
CudaUInt3_t from_cuda_uint3(uint3 value) noexcept {
    return CudaUInt3_t{value.x, value.y, value.z};
}
#endif

} // namespace orteaf::internal::backend::cuda
