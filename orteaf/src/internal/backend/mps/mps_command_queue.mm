#include "orteaf/internal/backend/mps/mps_command_queue.h"
#include "orteaf/internal/backend/mps/mps_stats.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#endif

namespace orteaf::internal::backend::mps {

MPSCommandQueue_t create_command_queue(MPSDevice_t device) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    stats_on_create_command_queue();
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    id<MTLCommandQueue> objc_command_queue = [objc_device newCommandQueue];
    return (MPSCommandQueue_t)opaque_from_objc_retained(objc_command_queue);
#else
    (void)device;
    return nullptr;
#endif
}

void destroy_command_queue(MPSCommandQueue_t command_queue) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (!command_queue) return;
    opaque_release_retained(command_queue);
    stats_on_destroy_command_queue();
#else
    (void)command_queue;
#endif
}

} // namespace orteaf::internal::backend::mps