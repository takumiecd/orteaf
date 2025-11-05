#include "orteaf/internal/backend/mps/mps_buffer.h"
#include "orteaf/internal/backend/mps/mps_stats.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

MPSBuffer_t create_buffer(MPSDevice_t device, size_t size, MPSBufferUsage_t usage) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    stats_on_create_buffer();
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    MTLResourceOptions objc_usage = static_cast<MTLResourceOptions>(usage);
    id<MTLBuffer> objc_buffer = [objc_device newBufferWithLength:size options:objc_usage];
    return (MPSBuffer_t)opaque_from_objc_retained(objc_buffer);
#else
    (void)device;
    (void)size;
    (void)usage;
    return nullptr;
#endif
}

void destroy_buffer(MPSBuffer_t buffer) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (buffer != nullptr) {
        opaque_release_retained(buffer);
        stats_on_destroy_buffer();
    }
#else
    (void)buffer;
#endif
}

const void* get_buffer_contents_const(MPSBuffer_t buffer) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!buffer) return nullptr;
    id<MTLBuffer> objc_buffer = objc_from_opaque_noown<id<MTLBuffer>>(buffer);
    return [objc_buffer contents];
#else
    (void)buffer;
    return nullptr;
#endif
}

void* get_buffer_contents(MPSBuffer_t buffer) {
    return const_cast<void*>(get_buffer_contents_const(buffer));
}
} // namespace orteaf::internal::backend::mps