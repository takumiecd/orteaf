/**
 * @file mps_buffer.mm
 * @brief Implementation of MPS/Metal buffer helpers.
 */
#include "orteaf/internal/backend/mps/mps_buffer.h"
#include "orteaf/internal/backend/mps/mps_stats.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "orteaf/internal/diagnostics/error/error.h"
#endif

namespace orteaf::internal::backend::mps {

/**
 * @copydoc orteaf::internal::backend::mps::createBuffer
 */
MPSBuffer_t createBuffer(MPSDevice_t device, size_t size, MPSBufferUsage_t usage) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (device == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throwError(OrteafErrc::NullPointer, "createBuffer: device cannot be nullptr");
    }
    if (size == 0) {
        using namespace orteaf::internal::diagnostics::error;
        throwError(OrteafErrc::InvalidParameter, "createBuffer: size cannot be 0");
    }
    id<MTLDevice> objc_device = objcFromOpaqueNoown<id<MTLDevice>>(device);
    MTLResourceOptions objc_usage = static_cast<MTLResourceOptions>(usage);
    id<MTLBuffer> objc_buffer = [objc_device newBufferWithLength:size options:objc_usage];
    updateAlloc(size);
    return (MPSBuffer_t)opaqueFromObjcRetained(objc_buffer);
#else
    (void)device;
    (void)size;
    (void)usage;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::destroyBuffer
 */
void destroyBuffer(MPSBuffer_t buffer) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (buffer != nullptr) {
        id<MTLBuffer> objc_buffer = objcFromOpaqueNoown<id<MTLBuffer>>(buffer);
        size_t size = objc_buffer ? [objc_buffer length] : 0;
        opaqueReleaseRetained(buffer);
        updateDealloc(size);
    }
#else
    (void)buffer;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::getBufferContentsConst
 */
const void* getBufferContentsConst(MPSBuffer_t buffer) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (!buffer) return nullptr;
    id<MTLBuffer> objc_buffer = objcFromOpaqueNoown<id<MTLBuffer>>(buffer);
    return [objc_buffer contents];
#else
    (void)buffer;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::getBufferContents
 */
void* getBufferContents(MPSBuffer_t buffer) {
    return const_cast<void*>(getBufferContentsConst(buffer));
}
} // namespace orteaf::internal::backend::mps