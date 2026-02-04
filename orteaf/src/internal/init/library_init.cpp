#include "orteaf/internal/init/library_init.h"

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>

namespace orteaf::internal::init {

namespace {

bool g_initialized = false;

}  // namespace

void initialize(const LibraryConfig &config) {
  if (g_initialized) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Library is already initialized");
  }

  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::configure(
      config.cpu_execution);
  ::orteaf::internal::tensor::api::TensorApi::configure(config.tensor_api);

  if (config.register_kernels) {
    ::orteaf::internal::kernel::registry::registerAllKernels();
  }

  g_initialized = true;
}

void shutdown() {
  if (!g_initialized) {
    return;
  }

  ::orteaf::internal::kernel::api::KernelRegistryApi::clear();
  ::orteaf::internal::tensor::api::TensorApi::shutdown();
  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::shutdown();
  ::orteaf::internal::execution_context::cpu::reset();

  g_initialized = false;
}

bool isInitialized() noexcept { return g_initialized; }

}  // namespace orteaf::internal::init
