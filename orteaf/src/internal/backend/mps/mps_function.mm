/**
 * @file mps_function.mm
 * @brief Implementation of MPS/Metal function helpers.
 */
#include "orteaf/internal/backend/mps/mps_function.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "orteaf/internal/diagnostics/error/error.h"
#endif

namespace orteaf::internal::backend::mps {

/**
 * @copydoc orteaf::internal::backend::mps::createFunction
 */
MPSFunction_t createFunction(MPSLibrary_t library, std::string_view name) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (library == nullptr) {
        using namespace orteaf::internal::diagnostics::error;
        throwError(OrteafErrc::NullPointer, "createFunction: library cannot be nullptr");
    }
    if (name.empty()) {
        using namespace orteaf::internal::diagnostics::error;
        throwError(OrteafErrc::InvalidParameter, "createFunction: name cannot be empty");
    }
    NSString* function_name = [[[NSString alloc] initWithBytes:name.data()
                                                   length:name.size()
                                                 encoding:NSUTF8StringEncoding] autorelease];
    if (function_name == nil) {
        return nullptr;
    }
    id<MTLLibrary> objc_library = objcFromOpaqueNoown<id<MTLLibrary>>(library);
    id<MTLFunction> objc_function = [objc_library newFunctionWithName:function_name];
    return (MPSFunction_t)opaqueFromObjcRetained(objc_function);
#else
    (void)library;
    (void)name;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::destroyFunction
 */
void destroyFunction(MPSFunction_t function) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (!function) return;
    opaqueReleaseRetained(function);
#else
    (void)function;
#endif
}

} // namespace orteaf::internal::backend::mps
