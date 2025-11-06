/**
 * @file mps_size.h
 * @brief POD size type and conversions for MPS/Metal.
 */
#pragma once

#include <cstdint>

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace orteaf::internal::backend::mps {

using MPSInt_t = int64_t; // 8 bytes size

/**
 * @brief 3D integer size (width/height/depth), ABI-stable 24 bytes.
 */
struct MPSSize_st {
    MPSInt_t width;
    MPSInt_t height;
    MPSInt_t depth;

    char padding[24 - 3 * sizeof(MPSInt_t)];
};

using MPSSize_t = MPSSize_st;

static_assert(sizeof(MPSSize_t) == 24, "MPSSize must be 24 bytes.");
static_assert(sizeof(MPSInt_t) == 8, "MPSInt must be 8 bytes.");

/** Construct an `MPSSize_t` from components. */
MPSSize_t make_size(MPSInt_t width, MPSInt_t height, MPSInt_t depth);

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
/** Convert to `MTLSize`. */
MTLSize to_mtl_size(MPSSize_t size);
/** Convert from `MTLSize`. */
MPSSize_t from_mtl_size(MTLSize mtl_size);
#endif

} // namespace orteaf::internal::backend::mps