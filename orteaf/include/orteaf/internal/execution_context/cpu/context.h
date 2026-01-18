#pragma once

#include "orteaf/internal/execution/cpu/manager/cpu_device_manager.h"

namespace orteaf::internal::execution_context::cpu {

class Context {
public:
  using DeviceLease =
      ::orteaf::internal::execution::cpu::manager::CpuDeviceManager::DeviceLease;

  DeviceLease device{};
};

} // namespace orteaf::internal::execution_context::cpu
