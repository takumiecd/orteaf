#pragma once

#include <cstddef>
#include <variant>

#include "orteaf/internal/backend/backend.h"
#include "orteaf/internal/runtime/allocator/buffer_resource.h"

namespace orteaf::internal::runtime::allocator {

// Type-erased wrapper around backend-specific BufferResource.
class Buffer {
  using CpuResource = BufferResource<::orteaf::internal::backend::Backend::Cpu>;
#if ORTEAF_ENABLE_CUDA
  using CudaResource =
      BufferResource<::orteaf::internal::backend::Backend::Cuda>;
#endif
#if ORTEAF_ENABLE_MPS
  using MpsResource = BufferResource<::orteaf::internal::backend::Backend::Mps>;
#endif

  using ResourceVariant = std::variant<
      CpuResource
#if ORTEAF_ENABLE_CUDA
      , CudaResource
#endif
#if ORTEAF_ENABLE_MPS
      , MpsResource
#endif
      >;

public:
  Buffer() = default;

  template <::orteaf::internal::backend::Backend B>
  explicit Buffer(const BufferResource<B> &res, std::size_t size_bytes = 0,
                  std::size_t alignment_bytes = 0)
      : backend_(B), resource_(res), size_(size_bytes),
        alignment_(alignment_bytes) {}

  ::orteaf::internal::backend::Backend backend() const noexcept {
    return backend_;
  }
  std::size_t size() const noexcept { return size_; }
  std::size_t alignment() const noexcept { return alignment_; }

  bool valid() const {
    return std::visit([](const auto &r) { return r.valid(); }, resource_);
  }

  template <::orteaf::internal::backend::Backend B>
  BufferResource<B> asResource() const {
    const auto *r = std::get_if<BufferResource<B>>(&resource_);
    return (r && backend_ == B) ? *r : BufferResource<B>{};
  }

  template <::orteaf::internal::backend::Backend B>
  explicit operator BufferResource<B>() const {
    return asResource<B>();
  }

private:
  ::orteaf::internal::backend::Backend backend_{
      ::orteaf::internal::backend::Backend::Cpu};
  ResourceVariant resource_{};
  std::size_t size_{0};
  std::size_t alignment_{0};
};

} // namespace orteaf::internal::runtime::allocator
