#pragma once

#include <utility>

#include "orteaf/internal/base/handle.h"
#include "orteaf/internal/runtime/base/backend_traits.h"

namespace orteaf::internal::runtime::allocator {

using BufferHandle = ::orteaf::internal::base::BufferHandle;

// Non-owning view of a buffer with an associated strong ID.
template <backend::Backend B> struct MemoryBlock {
  using BufferView = typename base::BackendTraits<B>::BufferView;

  BufferHandle id{};
  BufferView view{};

  MemoryBlock() = default;
  MemoryBlock(BufferHandle id, BufferView view)
      : id(id), view(std::move(view)) {}

  bool valid() const { return id.isValid() && static_cast<bool>(view); }
};

} // namespace orteaf::internal::runtime::allocator
