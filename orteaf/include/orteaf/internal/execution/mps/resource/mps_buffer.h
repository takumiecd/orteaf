#pragma once

#if ORTEAF_ENABLE_MPS

#include <utility>

#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/resource/mps_buffer_view.h"
#include "orteaf/internal/execution/mps/resource/mps_reuse_token.h"

namespace orteaf::internal::execution::mps::resource {

struct MpsBufferBlock {
  using BufferView = ::orteaf::internal::execution::mps::resource::MpsBufferView;
  using BufferViewHandle = ::orteaf::internal::execution::mps::MpsBufferViewHandle;

  BufferViewHandle handle{};
  BufferView view{};

  MpsBufferBlock() = default;
  MpsBufferBlock(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

struct MpsBuffer {
  using BufferView = MpsBufferBlock::BufferView;
  using BufferViewHandle = MpsBufferBlock::BufferViewHandle;
  using ReuseToken = ::orteaf::internal::execution::mps::resource::MpsReuseToken;
  using Block = MpsBufferBlock;

  BufferViewHandle handle{};
  BufferView view{};
  ReuseToken reuse_token{};

  MpsBuffer() = default;
  MpsBuffer(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  Block toBlock() const { return Block{handle, view}; }

  static MpsBuffer fromBlock(const Block &block) {
    return MpsBuffer{block.handle, block.view};
  }

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
