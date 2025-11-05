#include "orteaf/internal/backend/mps/mps_string.h"
#include "orteaf/internal/backend/mps/mps_objc_bridge.h"

#if defined(MPS_AVAILABLE) && defined(__OBJC__)
#import <Foundation/Foundation.h>
#endif

namespace orteaf::internal::backend::mps {

#if defined(MPS_AVAILABLE) && defined(__OBJC__)

MPSString_t to_ns_string(std::string_view view) {
    if (view.empty()) {
        return (MPSString_t)@"";
    }

    NSString* string = [[[NSString alloc] initWithBytes:view.data()
                                              length:view.size()
                                            encoding:NSUTF8StringEncoding] autorelease];
    if (string != nil) {
        return (MPSString_t)string;
    }

    NSString* fallback = [[[NSString alloc] initWithBytes:view.data()
                                                    length:view.size()
                                                  encoding:NSISOLatin1StringEncoding] autorelease];
    return (MPSString_t)fallback;
}

#else // !(defined(MPS_AVAILABLE) && defined(__OBJC__))

MPSString_t to_ns_string(std::string_view view) {
    (void)view;
    return nullptr;
}

#endif // defined(MPS_AVAILABLE) && defined(__OBJC__)

} // namespace orteaf::internal::backend::mps
