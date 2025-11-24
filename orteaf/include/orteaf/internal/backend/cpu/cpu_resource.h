#pragma once

#include <cstddef>
#include <sys/mman.h>
#include <unistd.h>

#include "orteaf/internal/backend/backend_traits.h"
#include "orteaf/internal/backend/cpu/cpu_buffer_view.h"
#include "orteaf/internal/backend/cpu/wrapper/cpu_alloc.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::backend::cpu {

// CPU backend resource used by allocator policies; non-owning BufferView wrapper.
struct CpuResource {
    using BufferView = ::orteaf::internal::backend::cpu::CpuBufferView;
    using Device = ::orteaf::internal::backend::BackendTraits<::orteaf::internal::backend::Backend::Cpu>::Device;
    using Context = ::orteaf::internal::backend::BackendTraits<::orteaf::internal::backend::Backend::Cpu>::Context;
    using Stream = ::orteaf::internal::backend::BackendTraits<::orteaf::internal::backend::Backend::Cpu>::Stream;

    // VA 予約。mmap で PROT_NONE の領域を確保し、PA は map で張る。
    static BufferView reserve(std::size_t size, Device /*device*/, Stream /*stream*/) {
        if (size == 0) {
            return {};
        }
        void* base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (base == MAP_FAILED) {
            diagnostics::error::throwError(diagnostics::error::OrteafErrc::OutOfMemory, "cpu reserve mmap failed");
        }
        return BufferView{base, 0, size};
    }

    static BufferView allocate(std::size_t size, std::size_t alignment, Device /*device*/, Stream /*stream*/) {
        if (size == 0) {
            return {};
        }
        void* base = cpu::allocAligned(size, alignment);
        return BufferView{base, 0, size};
    }

    static void deallocate(BufferView view, std::size_t size, std::size_t /*alignment*/, Device /*device*/, Stream /*stream*/) {
        if (!view) {
            return;
        }
        // Reconstruct the original base pointer from view data/offset.
        void* base = static_cast<void*>(static_cast<char*>(view.data()) - view.offset());
        cpu::dealloc(base, size);
    }

    // map/unmap は VA を RW に切り替え、unmap で解放する。
    static BufferView map(BufferView view, Device /*device*/, Context /*context*/, Stream /*stream*/) {
        if (!view) return view;
        void* base = reinterpret_cast<char*>(view.data()) - view.offset();
        if (mprotect(base, view.size(), PROT_READ | PROT_WRITE) != 0) {
            diagnostics::error::throwError(diagnostics::error::OrteafErrc::OperationFailed, "cpu map mprotect failed");
        }
        return view;
    }

    static void unmap(BufferView view, std::size_t size,
                      Device /*device*/, Context /*context*/, Stream /*stream*/) {
        if (!view) return;
        void* base = reinterpret_cast<char*>(view.data()) - view.offset();
        if (munmap(base, size) != 0) {
            diagnostics::error::throwError(diagnostics::error::OrteafErrc::OperationFailed, "cpu unmap munmap failed");
        }
    }
};

}  // namespace orteaf::internal::backend::cpu
