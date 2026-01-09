#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/manager/mps_fence_lifetime_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/platform/wrapper/mps_command_queue.h"

namespace orteaf::internal::execution::mps::resource {

class MpsCommandQueueResource {
public:
  using CommandQueueType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandQueue_t;
  using CommandQueueHandle =
      ::orteaf::internal::execution::mps::MpsCommandQueueHandle;
  using FenceLifetimeManager =
      ::orteaf::internal::execution::mps::manager::MpsFenceLifetimeManager;

  MpsCommandQueueResource() = default;

  CommandQueueType queue() const noexcept { return queue_; }
  bool hasQueue() const noexcept { return queue_ != nullptr; }
  void setQueue(CommandQueueType queue) noexcept { queue_ = queue; }

  FenceLifetimeManager &lifetime() noexcept { return lifetime_; }
  const FenceLifetimeManager &lifetime() const noexcept { return lifetime_; }

private:
  CommandQueueType queue_{nullptr};
  FenceLifetimeManager lifetime_{};
};

} // namespace orteaf::internal::execution::mps::resource

#endif // ORTEAF_ENABLE_MPS
