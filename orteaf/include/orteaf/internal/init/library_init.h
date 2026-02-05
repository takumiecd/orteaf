#pragma once

#include <orteaf/internal/execution/cpu/api/cpu_execution_api.h>
#include <orteaf/internal/tensor/api/tensor_api.h>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution/mps/api/mps_execution_api.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution/cuda/api/cuda_execution_api.h>
#endif

namespace orteaf::internal::init {

struct LibraryConfig {
  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::ExecutionManager::Config
      cpu_execution{};
#if ORTEAF_ENABLE_MPS
  ::orteaf::internal::execution::mps::api::MpsExecutionApi::ExecutionManager::Config
      mps_execution{};
#endif
#if ORTEAF_ENABLE_CUDA
  ::orteaf::internal::execution::cuda::api::CudaExecutionApi::ExecutionManager::Config
      cuda_execution{};
#endif
  ::orteaf::internal::tensor::api::TensorApi::Config tensor_api{};
  bool register_kernels = true;
};

void initialize(const LibraryConfig &config = {});
void shutdown();
bool isInitialized() noexcept;

}  // namespace orteaf::internal::init
