#include "orteaf/internal/backend/mps/mps_arch_detect.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

ARCH detect_mps_arch(MPSDevice_t device) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (device == nullptr) {
        return ARCH::MPS_Generic;
    }
    
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    if (objc_device == nil) {
        return ARCH::MPS_Generic;
    }
    
    // Check GPU family from newest to oldest
    // MPS_v3: Apple7+ (M3, M4, etc.)
    if ([objc_device supportsFamily:MTLGPUFamilyApple7]) {
        return ARCH::MPS_v3;
    }
    
    // MPS_v2: Apple4-6 (M1, M2, A14-A17)
    if ([objc_device supportsFamily:MTLGPUFamilyApple6] ||
        [objc_device supportsFamily:MTLGPUFamilyApple5] ||
        [objc_device supportsFamily:MTLGPUFamilyApple4]) {
        return ARCH::MPS_v2;
    }
    
    // MPS_v1: Apple1-3 (A12, A13, older)
    if ([objc_device supportsFamily:MTLGPUFamilyApple3] ||
        [objc_device supportsFamily:MTLGPUFamilyApple2] ||
        [objc_device supportsFamily:MTLGPUFamilyApple1]) {
        return ARCH::MPS_v1;
    }
    
    // Fallback to generic
    return ARCH::MPS_Generic;
#else
    (void)device;
    return ARCH::MPS_Generic;
#endif
}

} // namespace orteaf::internal::backend::mps
