#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/manager/mps_command_queue_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"

namespace orteaf::internal::execution_context::mps {

class Context {
public:
  using DeviceLease =
      ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease;
  using CommandQueueLease = ::orteaf::internal::execution::mps::manager::
      MpsCommandQueueManager::CommandQueueLease;

  DeviceLease device{};
  CommandQueueLease command_queue{};
};

} // namespace orteaf::internal::execution_context::mps

#endif // ORTEAF_ENABLE_MPS
