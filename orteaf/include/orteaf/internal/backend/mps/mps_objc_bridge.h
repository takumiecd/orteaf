// mps_objc_bridge.hpp  ← .mm だけが include する
#pragma once
#ifndef __OBJC__
#  error "This header must be included only from Objective-C++ (.mm)"
#endif
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

// 生キャスト（所有権は一切いじらない）
template <typename ObjcPtr>
static inline void* opaque_from_objc_noown(ObjcPtr p) noexcept {
    return (__bridge void*)p;
}
template <typename ObjcPtr>
static inline ObjcPtr objc_from_opaque_noown(void* p) noexcept {
    return (__bridge ObjcPtr)p;
}

// “返すハンドルは +1 で返す”ポリシーを採用するなら（非ARC前提）
template <typename ObjcPtr>
static inline void* opaque_from_objc_retained(ObjcPtr p) noexcept {
#if __has_feature(objc_arc)
    // ARCなら明示retain不要（保持したいなら strong 変数に保持）
    return (__bridge_retained void*)p;
#else
    [p retain];
    return (__bridge void*)p;
#endif
}

static inline void opaque_release_retained(void* p) noexcept {
#if __has_feature(objc_arc)
    CFBridgingRelease(p);
#else
    [(id)p release];
#endif
}

static inline void* opaque_retain(void* p) noexcept {
#if __has_feature(objc_arc)
    CFRetain(p);
#else
    [(id)p retain];
#endif
    return p;
}
