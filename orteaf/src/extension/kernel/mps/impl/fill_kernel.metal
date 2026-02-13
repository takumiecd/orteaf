// Fill float32 output view (supports contiguous and strided layouts).
#include <metal_stdlib>
using namespace metal;

constant uint kFillMaxRank = 8u;

struct FillLayoutParams {
  uint rank;
  uint shape[kFillMaxRank];
  int strides[kFillMaxRank];
};
// ABI contract: must match common::layout::FillLayoutParams on host side.

kernel void orteaf_fill_strided_f32(
    device float *out [[buffer(0)]],
    constant uint &offset [[buffer(1)]],
    constant uint &numel [[buffer(2)]],
    constant float &fill_value [[buffer(3)]],
    constant FillLayoutParams &layout [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < numel) {
    uint remaining = gid;
    long linear_index = static_cast<long>(offset);

    for (uint r = layout.rank; r > 0; --r) {
      const uint dim = r - 1u;
      const uint extent = layout.shape[dim];
      const uint coord = remaining % extent;
      remaining /= extent;
      linear_index += static_cast<long>(coord) *
                      static_cast<long>(layout.strides[dim]);
    }

    out[static_cast<uint>(linear_index)] = fill_value;
  }
}
