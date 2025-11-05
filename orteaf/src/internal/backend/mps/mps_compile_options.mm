#include "orteaf/internal/backend/mps/mps_compile_options.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

MPSCompileOptions_t create_compile_options() {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    return (MPSCompileOptions_t)opaque_from_objc_retained(options);
#else
    return nullptr;
#endif
}

void destroy_compile_options(MPSCompileOptions_t options) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (options != nullptr) {
        opaque_release_retained(options);
    }
#else
    (void)options;
#endif
}

void set_compile_options_math_mode(MPSCompileOptions_t options, bool fast_math_enabled) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (options == nullptr) return;
    
    MTLCompileOptions* objc_options = objc_from_opaque_noown<MTLCompileOptions*>(options);
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    objc_options.mathMode = fast_math_enabled ? MTLMathModeFast : MTLMathModeSafe;
#else
#error "macOS 15.0 SDK or later is required for MPS support"
#endif
#else
    (void)options;
    (void)fast_math_enabled;
#endif
}

void set_compile_options_preserve_invariance(MPSCompileOptions_t options, bool preserve_invariance) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (options == nullptr) return;
    
    MTLCompileOptions* objc_options = objc_from_opaque_noown<MTLCompileOptions*>(options);
    objc_options.preserveInvariance = preserve_invariance;
#else
    (void)options;
    (void)preserve_invariance;
#endif
}

void set_compile_options_preprocessor_macros(MPSCompileOptions_t options, void* macros_dictionary) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (options == nullptr) return;
    
    MTLCompileOptions* objc_options = objc_from_opaque_noown<MTLCompileOptions*>(options);
    NSDictionary* objc_dict = objc_from_opaque_noown<NSDictionary*>(macros_dictionary);
    objc_options.preprocessorMacros = objc_dict;
#else
    (void)options;
    (void)macros_dictionary;
#endif
}

} // namespace orteaf::internal::backend::mps
