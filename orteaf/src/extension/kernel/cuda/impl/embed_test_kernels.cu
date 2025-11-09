/**
 * @file embed_test_kernels.cu
 * @brief Sample CUDA kernels used to validate the kernel embedding pipeline.
 *
 * These kernels are intentionally trivial; they exist purely so the CMake
 * helpers can produce FATBIN/CUBIN/PTX blobs for tests to consume.
 */

#include <cuda_runtime.h>

extern "C" __global__ void orteaf_embed_test_fill_kernel(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

extern "C" __global__ void orteaf_embed_test_scale_kernel(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}
