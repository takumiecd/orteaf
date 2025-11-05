#include "orteaf/internal/backend/mps/mps_library.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dispatch/dispatch.h>
#endif

namespace orteaf::internal::backend::mps {

[[nodiscard]] MPSLibrary_t create_library(MPSDevice_t device, MPSString_t name, MPSError_t* error) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (name == nullptr) {
        if (error != nullptr) {
            *error = nullptr;
        }
        return nullptr;
    }

    NSString* library_name = objc_from_opaque_noown<NSString*>(name);
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    
    NSError* objc_error = nil;
    id<MTLLibrary> objc_library = nil;
    SEL selector = @selector(newLibraryWithName:error:);
    if ([objc_device respondsToSelector:selector]) {
        using NewLibraryFn = id<MTLLibrary> (*)(id, SEL, NSString*, NSError**);
        NewLibraryFn fn = reinterpret_cast<NewLibraryFn>(objc_msgSend);
        objc_library = fn(objc_device, selector, library_name, &objc_error);
    } else {
        objc_error = [NSError errorWithDomain:@"MTLLibraryErrorDomain"
                                        code:-2
                                    userInfo:@{NSLocalizedDescriptionKey : @"MTLDevice does not support newLibraryWithName:error:"}];
    }
    
    if (error != nullptr) {
        if (objc_error != nil) {
            *error = (MPSError_t)opaque_from_objc_retained(objc_error);
        } else {
            *error = nullptr;
        }
    }
    
    return (MPSLibrary_t)opaque_from_objc_retained(objc_library);
#else
    (void)device;
    (void)name;
    (void)error;
    return nullptr;
#endif
}

[[nodiscard]] MPSLibrary_t create_library_with_source(MPSDevice_t device,
                                                      MPSString_t source,
                                                      MPSCompileOptions_t compile_options,
                                                      MPSError_t* error) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (source == nullptr) {
        if (error != nullptr) {
            *error = nullptr;
        }
        return nullptr;
    }

    NSString* source_string = objc_from_opaque_noown<NSString*>(source);
    MTLCompileOptions* objc_compile_options = compile_options != nullptr
        ? objc_from_opaque_noown<MTLCompileOptions*>(compile_options)
        : nil;
    
    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    NSError* objc_error = nil;
    id<MTLLibrary> objc_library = [objc_device newLibraryWithSource:source_string options:objc_compile_options error:&objc_error];
    
    if (error != nullptr) {
        if (objc_error != nil) {
            *error = (MPSError_t)opaque_from_objc_retained(objc_error);
        } else {
            *error = nullptr;
        }
    }
    
    return (MPSLibrary_t)opaque_from_objc_retained(objc_library);
#else
    (void)device;
    (void)source;
    (void)compile_options;
    (void)error;
    return nullptr;
#endif
}

void destroy_library(MPSLibrary_t library) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (library != nullptr) {
        opaque_release_retained(library);
    }
#else
    (void)library;
#endif
}

[[nodiscard]] MPSLibrary_t create_library_with_data(MPSDevice_t device,
                                                    const void* data,
                                                    std::size_t size,
                                                    MPSError_t* error) {
#if defined(MPS_AVAILABLE) && defined(__OBJC__)
    if (data == nullptr || size == 0) {
        if (error != nullptr) {
            *error = nullptr;
        }
        return nullptr;
    }

    dispatch_data_t dispatch_data = dispatch_data_create(data, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (dispatch_data == nullptr) {
        if (error != nullptr) {
            *error = nullptr;
        }
        return nullptr;
    }

    id<MTLDevice> objc_device = objc_from_opaque_noown<id<MTLDevice>>(device);
    NSError* objc_error = nil;
    id<MTLLibrary> objc_library = nil;
    SEL selector = @selector(newLibraryWithData:error:);
    if ([objc_device respondsToSelector:selector]) {
        using NewLibraryWithDataFn = id<MTLLibrary> (*)(id, SEL, dispatch_data_t, NSError**);
        NewLibraryWithDataFn fn = reinterpret_cast<NewLibraryWithDataFn>(objc_msgSend);
        objc_library = fn(objc_device, selector, dispatch_data, &objc_error);
    } else {
        objc_error = [NSError errorWithDomain:@"MTLLibraryErrorDomain"
                                        code:-3
                                    userInfo:@{NSLocalizedDescriptionKey : @"MTLDevice does not support newLibraryWithData:error:"}];
    }
    dispatch_release(dispatch_data);

    if (error != nullptr) {
        if (objc_error != nil) {
            *error = (MPSError_t)opaque_from_objc_retained(objc_error);
        } else {
            *error = nullptr;
        }
    }
    
    return (MPSLibrary_t)opaque_from_objc_retained(objc_library);
#else
    (void)device;
    (void)data;
    (void)size;
    (void)error;
    return nullptr;
#endif
}

} // namespace orteaf::internal::backend::mps
