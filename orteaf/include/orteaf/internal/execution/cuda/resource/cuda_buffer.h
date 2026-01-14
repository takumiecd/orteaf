#pragma once

#if ORTEAF_ENABLE_CUDA

#include <utility>

#include "orteaf/internal/execution/cuda/cuda_handles.h"
#include "orteaf/internal/execution/cuda/resource/cuda_buffer_view.h"
#include "orteaf/internal/execution/cuda/resource/cuda_tokens.h"

namespace orteaf::internal::execution::cuda::resource {

struct CudaBufferBlock {
  using BufferView = ::orteaf::internal::execution::cuda::resource::CudaBufferView;
  using BufferViewHandle = ::orteaf::internal::execution::cuda::CudaBufferViewHandle;

  BufferViewHandle handle{};
  BufferView view{};

  CudaBufferBlock() = default;
  CudaBufferBlock(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

struct CudaBuffer {
  using BufferView = CudaBufferBlock::BufferView;
  using BufferViewHandle = CudaBufferBlock::BufferViewHandle;
  using ReuseToken = ::orteaf::internal::execution::cuda::resource::ReuseToken;
  using Block = CudaBufferBlock;

  BufferViewHandle handle{};
  BufferView view{};
  ReuseToken reuse_token{};

  CudaBuffer() = default;
  CudaBuffer(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  Block toBlock() const { return Block{handle, view}; }

  static CudaBuffer fromBlock(const Block &block) {
    return CudaBuffer{block.handle, block.view};
  }

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::execution::cuda::resource

#endif // ORTEAF_ENABLE_CUDA
