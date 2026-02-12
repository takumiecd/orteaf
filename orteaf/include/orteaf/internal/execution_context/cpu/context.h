#pragma once

#include "orteaf/internal/execution/cpu/cpu_handles.h"
#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"

namespace orteaf::internal::execution_context::cpu {

class Context {
public:
  using DeviceLease =
      ::orteaf::internal::execution::cpu::manager::CpuDeviceManager::DeviceLease;
  using Architecture = ::orteaf::internal::architecture::Architecture;

  /// @brief Create an empty context with no resources.
  Context() = default;

  /// @brief Create a context for the specified CPU device.
  /// @param device The device handle to create the context for.
  explicit Context(::orteaf::internal::execution::cpu::CpuDeviceHandle device);

  Architecture architecture() const noexcept {
    const auto *resource = device.operator->();
    if (resource == nullptr) {
      return Architecture::CpuGeneric;
    }
    return resource->arch;
  }

  DeviceLease device{};
};

} // namespace orteaf::internal::execution_context::cpu
