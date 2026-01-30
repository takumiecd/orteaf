#pragma once

#include "orteaf/internal/kernel/registry/kernel_registry.h"

namespace orteaf::internal::kernel::api {

/**
 * @brief Access the global KernelRegistry instance.
 */
::orteaf::internal::kernel::registry::KernelRegistry &kernelRegistry() noexcept;

} // namespace orteaf::internal::kernel::api
