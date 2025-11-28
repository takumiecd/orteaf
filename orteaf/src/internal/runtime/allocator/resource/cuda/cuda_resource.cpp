#include "orteaf/internal/runtime/allocator/resource/cuda/cuda_resource.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/backend/cuda/wrapper/cuda_alloc.h"

namespace orteaf::internal::backend::cuda {

void CudaResource::initialize(const Config& config) noexcept {
    config_ = config;
    initialized_ = true;
}

CudaResource::BufferView CudaResource::allocate(std::size_t size, std::size_t /*alignment*/) {
    if (size == 0) {
        return {};
    }

    CUdeviceptr_t base = ::orteaf::internal::backend::cuda::alloc(size);
    if (base == 0) {
        return {};
    }
    return BufferView{base, 0, size};
}

void CudaResource::deallocate(BufferView view, std::size_t size, std::size_t /*alignment*/) noexcept {
    if (!view) {
        return;
    }
    const auto base = view.data() - static_cast<CUdeviceptr_t>(view.offset());
    ::orteaf::internal::backend::cuda::free(base, size);
}

}  // namespace orteaf::internal::backend::cuda

#endif  // ORTEAF_ENABLE_CUDA
