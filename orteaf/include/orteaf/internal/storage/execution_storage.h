#pragma once

#include <utility>

#include <orteaf/internal/execution/allocator/execution_buffer.h>
#include <orteaf/internal/execution/base/execution_traits.h>

namespace orteaf::internal::storage {
template <execution::Execution B> class ExecutionStorage {
public:
  using Buffer = ::orteaf::internal::execution::allocator::ExecutionBuffer<B>;
  using FenceToken = ::orteaf::internal::execution::base::ExecutionTraits<B>::FenceToken;

  ExecutionStorage() = default;
  ExecutionStorage(Buffer buffer, FenceToken fence_token)
      : buffer_(std::move(buffer)), fence_token_(std::move(fence_token)) {}

  Buffer &buffer() { return buffer_; }
  const Buffer &buffer() const { return buffer_; }

  FenceToken &fenceToken() { return fence_token_; }
  const FenceToken &fenceToken() const { return fence_token_; }

  bool valid() const { return buffer_.valid(); }

private:
  Buffer buffer_{};
  FenceToken fence_token_{};
};
} // namespace orteaf::internal::storage
