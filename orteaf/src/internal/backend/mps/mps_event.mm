#if defined(MPS_AVAILABLE) && defined(__OBJC__)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "orteaf/internal/backend/mps/mps_event.h"
#include "orteaf/internal/backend/mps/mps_autorelease_pool.h"
#include "orteaf/internal/backend/mps/mps_command_buffer.h"
#include "orteaf/internal/backend/mps/mps_stats.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

namespace orteaf::internal::backend::mps {

MPSEvent_t create_event(MPSDevice_t device) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    orteaf::internal::backend::mps::AutoreleasePool pool{};
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    id<MTLSharedEvent> objc_event = [objc_device newSharedEvent];
    if (!objc_event) {
        return nullptr;
    }
    [objc_event setSignaledValue:0];
    stats_on_create_event();
    auto retained = opaque_from_objc_retained(objc_event);
    return (MPSEvent_t)retained;
#else
    (void)device;
    return nullptr;
#endif
}

void destroy_event(MPSEvent_t event) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!event) return;
    orteaf::internal::backend::mps::AutoreleasePool pool{};
    opaque_release_retained(event);
    stats_on_destroy_event();
#else
    (void)event;
#endif
}

void record_event(MPSEvent_t event, MPSCommandBuffer_t command_buffer, uint64_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!event) return;
    id<MTLSharedEvent> objc_event = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    if (command_buffer) {
        id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
        [objc_command_buffer encodeSignalEvent:objc_event value:value];
    } else {
        [objc_event setSignaledValue:value];
    }
#else
    (void)event;
    (void)command_buffer;
    (void)value;
#endif
}

bool query_event(MPSEvent_t event, uint64_t expected_value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!event) return true;
    id<MTLSharedEvent> objc_event = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    return [objc_event signaledValue] >= expected_value;
#else
    (void)event;
    (void)expected_value;
    return true;
#endif
}

uint64_t event_value(MPSEvent_t event) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!event) return 0;
    id<MTLSharedEvent> e = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    return [e signaledValue];
#else
    (void)event; return 0;
#endif
}

void write_event_queue(MPSCommandQueue_t command_queue, MPSEvent_t event, uint64_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    MPSCommandBuffer_t command_buffer = create_command_buffer(command_queue);
    record_event(event, command_buffer, value);
    commit(command_buffer);
    destroy_command_buffer(command_buffer);
#else
    (void)command_queue;
    (void)event;
    (void)value;
#endif
}

void wait_event(MPSCommandBuffer_t command_buffer, MPSEvent_t event, uint64_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!command_buffer || !event) return;
    id<MTLCommandBuffer> objc_command_buffer = objc_from_opaque_noown<id<MTLCommandBuffer>>(command_buffer);
    id<MTLSharedEvent> objc_event = objc_from_opaque_noown<id<MTLSharedEvent>>(event);
    [objc_command_buffer encodeWaitForEvent:objc_event value:value];
#else
    (void)command_buffer;
    (void)event;
    (void)value;
#endif
}

void wait_event_queue(MPSCommandQueue_t command_queue, MPSEvent_t event, uint64_t value) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    MPSCommandBuffer_t command_buffer = create_command_buffer(command_queue);
    wait_event(command_buffer, event, value);
    commit(command_buffer);
    destroy_command_buffer(command_buffer);
#else
    (void)command_queue;
    (void)event;
    (void)value;
#endif
}

} // namespace orteaf::internal::backend::mps

#endif // defined(MPS_AVAILABLE) && defined(__OBJC__)
