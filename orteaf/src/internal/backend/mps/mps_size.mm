#include "orteaf/internal/backend/mps/mps_size.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
static_assert(sizeof(MTLSize) == sizeof(MPSSize_t), "MTLSizeとMPSSize_tのサイズが一致する必要があります。");

#endif

namespace orteaf::internal::backend::mps {

MPSSize_t make_size(MPSInt_t width, MPSInt_t height, MPSInt_t depth) {
    MPSSize_t size{};
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    size.width = static_cast<MPSInt_t>(width);
    size.height = static_cast<MPSInt_t>(height);
    size.depth = static_cast<MPSInt_t>(depth);
#else
    size.width = width;
    size.height = height;
    size.depth = depth;
#endif
    return size;
}

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
MTLSize to_mtl_size(MPSSize_t size) {
    return MTLSizeMake(static_cast<NSUInteger>(size.width),
                       static_cast<NSUInteger>(size.height),
                       static_cast<NSUInteger>(size.depth));
}

MPSSize_t from_mtl_size(MTLSize mtl_size) {
    MPSSize_t out{};
    out.width = static_cast<MPSInt_t>(mtl_size.width);
    out.height = static_cast<MPSInt_t>(mtl_size.height);
    out.depth = static_cast<MPSInt_t>(mtl_size.depth);
    return out;
}
#endif

} // namespace orteaf::internal::backend::mps