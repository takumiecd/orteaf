#include "orteaf/internal/backend/mps/mps_device.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

MPSDevice_t get_device() {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return nullptr;
    }
    return (MPSDevice_t)opaque_from_objc_retained(device);
#else
    return nullptr;
#endif
}

MPSDevice_t get_device(MPSInt_t device_id) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices == nil) {
        return nullptr;
    }
    if (device_id < 0) {
        [devices release];
        return nullptr;
    }
    NSUInteger index = static_cast<NSUInteger>(device_id);
    if (index >= [devices count]) {
        [devices release];
        return nullptr;
    }
    id<MTLDevice> device = [devices objectAtIndex:index];
    MPSDevice_t handle = (MPSDevice_t)opaque_from_objc_retained(device);
    [devices release];
    return handle;
#else
    (void)device_id;
    return nullptr;
#endif
}

int get_device_count() {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    NSUInteger count = devices != nil ? [devices count] : 0;
    if (devices != nil) {
        [devices release];
    }
    return static_cast<int>(count);
#else
    return 0;
#endif
}

void device_retain(MPSDevice_t device) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (device == nullptr) {
        return;
    }
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    [objc_device retain];
#else
    (void)device;
#endif
}

void device_release(MPSDevice_t device) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (device == nullptr) {
        return;
    }
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    [objc_device release];
#else
    (void)device;
#endif
}

MPSDeviceArray_t get_device_array() {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    return (MPSDeviceArray_t)opaque_from_objc_noown(devices);
#else
    return nullptr;
#endif
}

} // namespace orteaf::internal::backend::mps
