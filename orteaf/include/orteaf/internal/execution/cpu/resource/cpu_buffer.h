#pragma once

#include <utility>

#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/resource/cpu_buffer_view.h"
#include "orteaf/internal/execution/cpu/resource/cpu_tokens.h"

namespace orteaf::internal::execution::cpu::resource {

struct CpuBufferBlock {
  using BufferView = ::orteaf::internal::execution::cpu::resource::CpuBufferView;
  using BufferViewHandle = ::orteaf::internal::execution::cpu::CpuBufferViewHandle;

  BufferViewHandle handle{};
  BufferView view{};

  CpuBufferBlock() = default;
  CpuBufferBlock(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

struct CpuBuffer {
  using BufferView = CpuBufferBlock::BufferView;
  using BufferViewHandle = CpuBufferBlock::BufferViewHandle;
  using ReuseToken = ::orteaf::internal::execution::cpu::resource::ReuseToken;
  using Block = CpuBufferBlock;

  BufferViewHandle handle{};
  BufferView view{};
  ReuseToken reuse_token{};

  CpuBuffer() = default;
  CpuBuffer(BufferViewHandle h, BufferView v)
      : handle(h), view(std::move(v)) {}

  Block toBlock() const { return Block{handle, view}; }

  static CpuBuffer fromBlock(const Block &block) {
    return CpuBuffer{block.handle, block.view};
  }

  bool valid() const { return handle.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::execution::cpu::resource
