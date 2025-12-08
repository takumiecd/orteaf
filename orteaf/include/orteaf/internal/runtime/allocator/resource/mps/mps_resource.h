#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>

#include "orteaf/internal/runtime/mps/resource/mps_buffer_view.h"
#include "orteaf/internal/runtime/mps/resource/mps_reuse_token.h"
#include "orteaf/internal/runtime/mps/resource/mps_fence_token.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_buffer.h"
#include "orteaf/internal/runtime/mps/platform/wrapper/mps_heap.h"
#include <orteaf/internal/backend/backend_traits.h>

namespace orteaf::internal::backend::mps {

// Simple MPS resource that keeps device/heap handles per instance and creates buffers at offset 0.
class MpsResource {
public:
    using BufferView = ::orteaf::internal::runtime::mps::resource::MpsBufferView;
    using FenceToken = ::orteaf::internal::runtime::mps::resource::MpsFenceToken;
    using ReuseToken = ::orteaf::internal::runtime::mps::resource::MpsReuseToken;

    struct Config {
        ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device{nullptr};
        ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap{nullptr};
        ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t usage{::orteaf::internal::runtime::mps::platform::wrapper::kMPSDefaultBufferUsage};
    };

    MpsResource() = default;

    explicit MpsResource(const Config& config) { initialize(config); }

    void initialize(const Config& config);

    BufferView allocate(std::size_t size, std::size_t alignment);

    void deallocate(BufferView view, std::size_t size, std::size_t alignment) noexcept;

    ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device() const noexcept { return device_; }
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap() const noexcept { return heap_; }

    bool isCompleted(FenceToken& token);
    bool isCompleted(ReuseToken& token);

    static BufferView makeView(BufferView base, std::size_t offset, std::size_t size);

private:
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSDevice_t device_{nullptr};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t heap_{nullptr};
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSBufferUsage_t usage_{::orteaf::internal::runtime::mps::platform::wrapper::kMPSDefaultBufferUsage};
    bool initialized_{false};
};

}  // namespace orteaf::internal::backend::mps

#endif  // ORTEAF_ENABLE_MPS
