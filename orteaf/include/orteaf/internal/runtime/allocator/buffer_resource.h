#pragma once

#include <utility>

#include <orteaf/internal/backend/backend.h>
#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/runtime/cpu/resource/cpu_buffer_view.h>

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/runtime/cuda/resource/cuda_buffer_view.h>
#endif // ORTEAF_ENABLE_CUDA

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/runtime/mps/resource/mps_buffer_view.h>
#include <orteaf/internal/runtime/mps/resource/mps_fence_token.h>
#endif // ORTEAF_ENABLE_MPS

namespace orteaf::internal::runtime::allocator {
template <backend::Backend B>
struct ResourceBufferType {
  using view = ::orteaf::internal::runtime::cpu::resource::CpuBufferView;
};

#if ORTEAF_ENABLE_CUDA
template <> struct ResourceBufferType<backend::Backend::Cuda> {
  using view = ::orteaf::internal::runtime::cuda::resource::CudaBufferView;
};
#endif // ORTEAF_ENABLE_CUDA

#if ORTEAF_ENABLE_MPS
template <> struct ResourceBufferType<backend::Backend::Mps> {
  using view = ::orteaf::internal::runtime::mps::resource::MpsBufferView;
  using fence_token =
      ::orteaf::internal::runtime::mps::resource::MpsFenceToken;
};
#endif // ORTEAF_ENABLE_MPS

// Non-owning view of a buffer with an associated strong ID.
template <backend::Backend B> struct BufferResource {
  using BufferView = typename ResourceBufferType<B>::view;
  using BufferViewHandle = ::orteaf::internal::base::BufferViewHandle;
  using FenceToken = typename ResourceBufferType<B>::fence_token;

  BufferViewHandle handle{};
  BufferView view{};
  FenceToken fence_token{};

  BufferResource() = default;
  BufferResource(BufferViewHandle handle, BufferView view)
      : handle(handle), view(std::move(view)) {}

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::runtime::allocator
