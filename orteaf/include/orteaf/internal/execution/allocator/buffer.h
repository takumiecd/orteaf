#pragma once

#include <utility>

#include <orteaf/internal/base/handle.h>
#include <orteaf/internal/execution/cpu/resource/cpu_buffer_view.h>
#include <orteaf/internal/execution/execution.h>

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution/cuda/resource/cuda_buffer_view.h>
#endif // ORTEAF_ENABLE_CUDA

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/resource/mps_buffer_view.h>
#include <orteaf/internal/execution/mps/resource/mps_reuse_token.h>
#endif // ORTEAF_ENABLE_MPS

namespace orteaf::internal::execution::allocator {
struct CpuReuseToken {};

template <execution::Execution B> struct ResourceBufferType {
  using view = ::orteaf::internal::execution::cpu::resource::CpuBufferView;
  using reuse_token = CpuReuseToken;
};

#if ORTEAF_ENABLE_CUDA
template <> struct ResourceBufferType<execution::Execution::Cuda> {
  using view = ::orteaf::internal::execution::cuda::resource::CudaBufferView;
};
#endif // ORTEAF_ENABLE_CUDA

#if ORTEAF_ENABLE_MPS
template <> struct ResourceBufferType<execution::Execution::Mps> {
  using view = ::orteaf::internal::execution::mps::resource::MpsBufferView;
  using reuse_token =
      ::orteaf::internal::execution::mps::resource::MpsReuseToken;
};
#endif // ORTEAF_ENABLE_MPS

// Lightweight pair of buffer view and handle (no reuse tracking).
template <execution::Execution B> struct ExecutionBufferBlock {
  using BufferView = typename ResourceBufferType<B>::view;
  using BufferViewHandle = ::orteaf::internal::base::BufferViewHandle;

  BufferViewHandle handle{};
  BufferView view{};

  ExecutionBufferBlock() = default;
  ExecutionBufferBlock(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

// Non-owning view of a buffer with an associated strong ID.
template <execution::Execution B> struct ExecutionBuffer {
  using BufferView = typename ResourceBufferType<B>::view;
  using BufferViewHandle = ::orteaf::internal::base::BufferViewHandle;
  using ReuseToken = typename ResourceBufferType<B>::reuse_token;

  BufferViewHandle handle{};
  BufferView view{};
  ReuseToken reuse_token{};

  ExecutionBuffer() = default;
  ExecutionBuffer(BufferViewHandle handle, BufferView view)
      : handle(handle), view(std::move(view)) {}

  // Convert to ExecutionBufferBlock (discards reuse_token)
  ExecutionBufferBlock<B> toBlock() const {
    return ExecutionBufferBlock<B>{handle, view};
  }

  // Construct from ExecutionBufferBlock (reuse_token is default-initialized)
  static ExecutionBuffer fromBlock(const ExecutionBufferBlock<B> &block) {
    return ExecutionBuffer{block.handle, block.view};
  }

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::execution::allocator
