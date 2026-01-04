#pragma once

#include <orteaf/internal/execution/allocator/resource/mps/mps_resource.h>
#include <orteaf/internal/execution/mps/manager/mps_buffer_manager.h>
#include <orteaf/internal/execution/mps/manager/mps_fence_lifetime_manager.h>

namespace orteaf::internal::storage::mps {
template <typename B> class MpsStorage {
public:
  using MpsResource = ::orteaf::internal::execution::allocator::resource::mps::MpsResource;
  using StrongBufferLease =
      ::orteaf::internal::execution::mps::manager::MpsBufferManager<MpsResource>::StrongBufferLease;
  using WeakBufferLease =
      ::orteaf::internal::execution::mps::manager::MpsBufferManager<MpsResource>::WeakBufferLease;
  using FenceToken = ::orteaf::internal::execution::mps::manager::MpsFenceLifetimeManager::

private:
    StrongBufferLease strong_buffer_lease_;

};
} // namespace orteaf::internal::storage::mps
