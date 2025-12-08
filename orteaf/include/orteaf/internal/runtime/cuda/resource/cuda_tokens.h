#pragma once

namespace orteaf::internal::runtime::cuda::resource {

struct FenceToken {
  void *value;
};

struct ReuseToken {
  void *value;
};

} // namespace orteaf::internal::runtime::cuda::resource
