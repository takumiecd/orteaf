/**
 * @file mps_size.mm
 * @brief Implementation for MPS size helpers and Metal conversions.
 */
#include "orteaf/internal/backend/mps/mps_size.h"

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace orteaf::internal::backend::mps {

/**
 * @copydoc orteaf::internal::backend::mps::make_size
 */
MPSSize_t make_size(MPSInt_t width, MPSInt_t height, MPSInt_t depth) {
    MPSSize_t size{};
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
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

/** Convert to `MTLSize`. */
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
MTLSize to_mtl_size(MPSSize_t size) {
    return MTLSizeMake(static_cast<NSUInteger>(size.width),
                       static_cast<NSUInteger>(size.height),
                       static_cast<NSUInteger>(size.depth));
}

/** Convert from `MTLSize`. */
MPSSize_t from_mtl_size(MTLSize mtl_size) {
    MPSSize_t out{};
    out.width = static_cast<MPSInt_t>(mtl_size.width);
    out.height = static_cast<MPSInt_t>(mtl_size.height);
    out.depth = static_cast<MPSInt_t>(mtl_size.depth);
    return out;
}
#endif

} // namespace orteaf::internal::backend::mps