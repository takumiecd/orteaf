// Sample Metal kernel library used to ensure the embedding pipeline works.
#include <metal_stdlib>
using namespace metal;

kernel void orteaf_embed_test_identity(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < length) {
        // No-op kernel; keeps value unchanged to verify we can load the library.
        data[gid] = data[gid];
    }
}
