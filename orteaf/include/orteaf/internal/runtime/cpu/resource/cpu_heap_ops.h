#pragma once

#include <cstddef>

#include <orteaf/internal/runtime/cpu/resource/cpu_buffer_view.h>
#include <orteaf/internal/runtime/cpu/resource/cpu_heap_region.h>

namespace orteaf::internal::runtime::cpu::resource {

// Low-level heap operations for CPU backend.
// Used by HierarchicalSlotAllocator for VA reservation and mapping.
struct CpuHeapOps {
    using BufferView = ::orteaf::internal::runtime::cpu::resource::CpuBufferView;
    using HeapRegion = ::orteaf::internal::runtime::cpu::resource::CpuHeapRegion;

    // VA reservation. Allocates PROT_NONE region via mmap.
    static HeapRegion reserve(std::size_t size);

    // Map reserved region to RW.
    static BufferView map(HeapRegion region);

    // Unmap and release the region.
    static void unmap(HeapRegion region, std::size_t size);
};

}  // namespace orteaf::internal::runtime::cpu::resource