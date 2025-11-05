#pragma once

#include <cstdint>

using MPSInt_t = int64_t; // 8 bytes size

struct MPSSize_st {
    MPSInt_t width;
    MPSInt_t height;
    MPSInt_t depth;

    char padding[24 - 3 * sizeof(MPSInt_t)];
};

using MPSSize_t = MPSSize_st;

static_assert(sizeof(MPSSize_t) == 24, "MPSSize must be 24 bytes.");
static_assert(sizeof(MPSInt_t) == 8, "MPSInt must be 8 bytes.");

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace orteaf::internal::backend::mps {

MPSSize_t make_size(MPSInt_t width, MPSInt_t height, MPSInt_t depth);

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
MTLSize to_mtl_size(MPSSize_t size);
MPSSize_t from_mtl_size(MTLSize mtl_size);
#endif

} // namespace orteaf::internal::backend::mps