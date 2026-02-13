#include <metal_stdlib>

using namespace metal;

struct TransferLayoutParams {
  uint rank;
  uint shape[8];
  int src_strides[8];
  int dst_strides[8];
};
// ABI contract: must match common::layout::TransferLayoutParams on host side.

inline long linearToStridedOffset(uint linear, uint rank,
                                  constant uint *shape,
                                  constant int *strides) {
  ulong remaining = static_cast<ulong>(linear);
  long offset = 0;
  for (uint dim = rank; dim > 0; --dim) {
    const uint i = dim - 1;
    const ulong dim_size = static_cast<ulong>(shape[i]);
    const ulong coord = dim_size == 0 ? 0 : (remaining % dim_size);
    remaining = dim_size == 0 ? 0 : (remaining / dim_size);
    offset += static_cast<long>(coord) * static_cast<long>(strides[i]);
  }
  return offset;
}

kernel void orteaf_copy_contiguous_to_strided_u8(
    device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
    constant uint &src_offset [[buffer(2)]],
    constant uint &dst_offset [[buffer(3)]], constant uint &numel [[buffer(4)]],
    constant uint &elem_size [[buffer(5)]],
    constant TransferLayoutParams &layout [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= numel) {
    return;
  }
  const uint src_elem = src_offset + gid;
  const long dst_elem_signed =
      static_cast<long>(dst_offset) +
      linearToStridedOffset(gid, layout.rank, layout.shape,
                            layout.dst_strides);
  const uint dst_elem = static_cast<uint>(dst_elem_signed);
  const uint src_byte = src_elem * elem_size;
  const uint dst_byte = dst_elem * elem_size;
  for (uint i = 0; i < elem_size; ++i) {
    dst[dst_byte + i] = src[src_byte + i];
  }
}

kernel void orteaf_copy_strided_to_contiguous_u8(
    device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
    constant uint &src_offset [[buffer(2)]],
    constant uint &dst_offset [[buffer(3)]], constant uint &numel [[buffer(4)]],
    constant uint &elem_size [[buffer(5)]],
    constant TransferLayoutParams &layout [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= numel) {
    return;
  }
  const long src_elem_signed =
      static_cast<long>(src_offset) +
      linearToStridedOffset(gid, layout.rank, layout.shape,
                            layout.src_strides);
  const uint src_elem = static_cast<uint>(src_elem_signed);
  const uint dst_elem = dst_offset + gid;
  const uint src_byte = src_elem * elem_size;
  const uint dst_byte = dst_elem * elem_size;
  for (uint i = 0; i < elem_size; ++i) {
    dst[dst_byte + i] = src[src_byte + i];
  }
}
