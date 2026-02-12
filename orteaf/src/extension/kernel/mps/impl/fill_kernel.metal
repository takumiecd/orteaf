// Fill contiguous float32 output view.
#include <metal_stdlib>
using namespace metal;

kernel void orteaf_fill_contiguous_f32(
    device float *out [[buffer(0)]],
    constant uint &offset [[buffer(1)]],
    constant uint &numel [[buffer(2)]],
    constant float &fill_value [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < numel) {
    out[offset + gid] = fill_value;
  }
}
