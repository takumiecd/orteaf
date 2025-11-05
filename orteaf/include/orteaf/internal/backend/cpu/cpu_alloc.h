#pragma once

#include "orteaf/internal/backend/cpu/cpu_stats.h"

#include <cstdlib>
#include <new>
#include <cstdint>

namespace orteaf::internal::backend::cpu {

constexpr std::size_t kCpuDefaultAlign = alignof(std::max_align_t);

inline bool is_pow2(std::size_t x) { return x && ((x & (x-1))==0); }
inline std::size_t next_pow2(std::size_t x){
    if (x<=1) return 1u;
    --x; x|=x>>1; x|=x>>2; x|=x>>4; x|=x>>8; x|=x>>16;
    if constexpr (sizeof(std::size_t)==8) x|=x>>32;
    return x+1;
}

inline void* alloc(std::size_t size) {
    return alloc_aligned(size, kCpuDefaultAlign);
}

inline void* alloc_aligned(std::size_t size, std::size_t alignment) {
    if (size == 0) return nullptr;

    const std::size_t min_align = alignof(std::max_align_t);
    if (alignment < min_align) alignment = min_align;
    if (!is_pow2(alignment)) alignment = next_pow2(alignment);

#if defined(_MSC_VER)
    void* p = _aligned_malloc(size, alignment);
    if (!p) throw std::bad_alloc();
#else
    // aligned_alloc は size が alignment の倍数要件あり→posix_memalign優先
    void* p = nullptr;
    const int rc = ::posix_memalign(&p, alignment, size);
    if (rc != 0 || !p) throw std::bad_alloc();
#endif

    update_alloc(size);
    return p;
}

inline void dealloc(void* ptr, std::size_t size) noexcept {
    if (!ptr) return;
    update_dealloc(size);
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

} // namespace orteaf::internal::backend::cpu