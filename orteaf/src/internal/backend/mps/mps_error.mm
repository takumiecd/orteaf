/**
 * @file mps_error.mm
 * @brief Implementation of NSError construction/destruction helpers.
 */
#include "orteaf/internal/backend/mps/mps_error.h"
#include "orteaf/internal/backend/mps/mps_string.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)

namespace {

/** Internal helper to construct NSError with optional userInfo. */
[[nodiscard]] MPSError_t make_error(std::string_view domain,
                                    std::string_view description,
                                    NSDictionary* additional_user_info = nil) {
    MPSString_t domain_string_opaque = to_ns_string(domain);
    NSString* domain_string = objc_from_opaque_noown<NSString*>(domain_string_opaque);
    if (domain_string == nil || domain_string.length == 0) {
        domain_string = @"orteaf.mps";
    }

    MPSString_t description_string_opaque = to_ns_string(description);
    NSString* description_string = objc_from_opaque_noown<NSString*>(description_string_opaque);
    NSMutableDictionary* user_info = [[[NSMutableDictionary alloc] init] autorelease];
    if (description_string != nil && description_string.length > 0) {
        user_info[NSLocalizedDescriptionKey] = description_string;
    }
    if (additional_user_info != nil) {
        [user_info addEntriesFromDictionary:additional_user_info];
    }

    NSError* objc_error = [[NSError alloc] initWithDomain:domain_string code:0 userInfo:user_info.count > 0 ? user_info : nil];
    return (MPSError_t)opaque_from_objc_retained(objc_error);
}

} // namespace

#endif // defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)

/**
 * @copydoc orteaf::internal::backend::mps::create_error(const std::string&)
 */
MPSError_t create_error(const std::string& message) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    return make_error("NSCocoaErrorDomain", message);
#else
    (void)message;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::create_error(std::string_view,std::string_view)
 */
MPSError_t create_error(std::string_view domain, std::string_view description) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    return make_error(domain, description);
#else
    (void)domain;
    (void)description;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::create_error(std::string_view,std::string_view,void*)
 */
MPSError_t create_error(std::string_view domain,
                        std::string_view description,
                        void* additional_user_info) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    NSDictionary* objc_user_info = objc_from_opaque_noown<NSDictionary*>(additional_user_info);
    return make_error(domain, description, objc_user_info);
#else
    (void)domain;
    (void)description;
    (void)additional_user_info;
    return nullptr;
#endif
}

/**
 * @copydoc orteaf::internal::backend::mps::destroy_error
 */
void destroy_error(MPSError_t error) {
#if defined(ORTEAF_ENABLE_MPS) && defined(__OBJC__)
    if (error != nullptr) {
        opaque_release_retained(error);
    }
#else
    (void)error;
#endif
}

} // namespace orteaf::internal::backend::mps