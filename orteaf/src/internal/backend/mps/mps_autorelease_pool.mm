#include "orteaf/internal/backend/mps/mps_autorelease_pool.h"

#ifdef MPS_AVAILABLE

#import <Foundation/Foundation.h>
#include <chrono>
#include <iostream>

namespace orteaf::internal::backend::mps {

AutoreleasePool::AutoreleasePool() {
#if __has_feature(objc_arc)
    // ARCでもNSAutoreleasePoolは使える（推奨は@autoreleasepoolだが、ラッパー内なら可）
    pool_ = (NSAutoreleasePool*)[[NSAutoreleasePool alloc] init];
#else
    pool_ = [[NSAutoreleasePool alloc] init];
#endif
}

AutoreleasePool::~AutoreleasePool() {
    if (pool_) {
#if __has_feature(objc_arc)
        [(id)pool_ drain];
#else
        [pool_ drain];
#endif
        pool_ = nullptr;
    }
}

} // namespace orteaf::internal::backend::mps

#endif // MPS_AVAILABLE


