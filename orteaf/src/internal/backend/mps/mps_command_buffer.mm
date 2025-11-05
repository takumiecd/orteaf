#include "orteaf/internal/backend/mps/mps_command_buffer.h"
#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_event.h"
#include "orteaf/internal/backend/mps/mps_autorelease_pool.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace orteaf::internal::backend::mps {

using orteaf::internal::backend::mps::AutoreleasePool;

MPSCommandBuffer_t create_command_buffer(MPSCommandQueue_t command_queue) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    AutoreleasePool pool{};
    id<MTLCommandQueue> objc_command_queue = objc_from_opaque_noown<id<MTLCommandQueue>>(command_queue);
    id<MTLCommandBuffer> objc_command_buffer = [objc_command_queue commandBuffer];
    return (MPSCommandBuffer_t)opaque_from_objc_retained(objc_command_buffer);
#else
    (void)command_queue;
    return nullptr;
#endif
}

void destroy_command_buffer(MPSCommandBuffer_t command_buffer) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (command_buffer != nullptr) {
        AutoreleasePool pool{};
        opaque_release_retained(command_buffer);
    }
#else
    (void)command_buffer;
#endif
}

void encode_signal_event(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
    id<MTLSharedEvent> objc_event = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    [objc_command_buffer encodeSignalEvent:objc_event value:value];
#else
    (void)command_buffer;
    (void)event;
    (void)value;
#endif
}

void encode_wait(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint32_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
    id<MTLSharedEvent> objc_event = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    [objc_command_buffer encodeWaitForEvent:objc_event value:value];
#else
    (void)command_buffer;
    (void)event;
    (void)value;
#endif
}

void commit(MPSCommandBuffer_t command_buffer) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
    [objc_command_buffer commit];
#else
    (void)command_buffer;
#endif
}

void wait_until_completed(MPSCommandBuffer_t command_buffer) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
    [objc_command_buffer waitUntilCompleted];
#else
    (void)command_buffer;
#endif
}

} // namespace orteaf::internal::backend::mps
