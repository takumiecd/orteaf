#pragma once

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/manager/mps_command_queue_manager.h"
#include "orteaf/internal/execution/mps/manager/mps_device_manager.h"

namespace orteaf::internal::execution_context::mps {

class Context {
public:
  using DeviceLease =
      ::orteaf::internal::execution::mps::manager::MpsDeviceManager::DeviceLease;
  using Architecture = ::orteaf::internal::architecture::Architecture;
  using CommandQueueLease = ::orteaf::internal::execution::mps::manager::
      MpsCommandQueueManager::CommandQueueLease;

  /// @brief Create an empty context with no resources.
  Context() = default;

  /// @brief Create a context for the specified device with a new command queue.
  /// @param device The device handle to create the context for.
  explicit Context(::orteaf::internal::execution::mps::MpsDeviceHandle device);

  /// @brief Create a context for the specified device and command queue.
  /// @param device The device handle to create the context for.
  /// @param command_queue The command queue handle to acquire.
  Context(::orteaf::internal::execution::mps::MpsDeviceHandle device,
          ::orteaf::internal::execution::mps::MpsCommandQueueHandle command_queue);

  Architecture architecture() const noexcept {
    const auto *resource = device.operator->();
    if (resource == nullptr) {
      return Architecture::MpsGeneric;
    }
    return resource->architecture();
  }

  DeviceLease device{};
  CommandQueueLease command_queue{};
};

} // namespace orteaf::internal::execution_context::mps

#endif // ORTEAF_ENABLE_MPS
