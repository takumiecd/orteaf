#pragma once

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/tensor/api/tensor_api.h>

namespace orteaf::internal::init {

struct LibraryConfig {
  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::ExecutionManager::Config
      cpu_execution{};
  ::orteaf::internal::tensor::api::TensorApi::Config tensor_api{};
  bool register_kernels = true;
};

void initialize(const LibraryConfig &config = {});
void shutdown();
bool isInitialized() noexcept;

}  // namespace orteaf::internal::init
