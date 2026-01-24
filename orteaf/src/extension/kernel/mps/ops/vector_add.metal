// Vector Add Metal Kernel
// C = A + B
#include <metal_stdlib>
using namespace metal;

kernel void orteaf_vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < num_elements) {
        c[gid] = a[gid] + b[gid];
    }
}
