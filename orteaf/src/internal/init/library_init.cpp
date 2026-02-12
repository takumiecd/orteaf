#include "orteaf/internal/init/library_init.h"

#include <orteaf/internal/diagnostics/error/error.h>
#include <orteaf/internal/execution_context/cpu/current_context.h>
#include <orteaf/internal/kernel/api/kernel_registry_api.h>
#include <orteaf/internal/kernel/registry/kernel_auto_registry.h>

#include <atomic>
#include <mutex>

#if ORTEAF_ENABLE_MPS
#include <orteaf/internal/execution_context/mps/current_context.h>
#include <orteaf/internal/execution/mps/platform/wrapper/mps_device.h>
#endif

#if ORTEAF_ENABLE_CUDA
#include <orteaf/internal/execution_context/cuda/current_context.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_device.h>
#include <orteaf/internal/execution/cuda/platform/wrapper/cuda_init.h>
#endif

namespace orteaf::internal::init {

namespace {

std::atomic<bool> g_initialized{false};
std::mutex g_init_mutex;

template <typename PoolConfig>
void ensurePoolConfig(PoolConfig &cfg, std::size_t capacity) {
  if (capacity == 0) {
    return;
  }
  if (cfg.control_block_capacity == 0) {
    cfg.control_block_capacity = capacity;
  }
  if (cfg.control_block_block_size == 0) {
    cfg.control_block_block_size = cfg.control_block_capacity;
  }
  if (cfg.payload_capacity == 0) {
    cfg.payload_capacity = capacity;
  }
  if (cfg.payload_block_size == 0) {
    cfg.payload_block_size = cfg.payload_capacity;
  }
}

#if ORTEAF_ENABLE_MPS
void applyMpsDefaults(LibraryConfig &config) {
  auto &exec_config = config.mps_execution;
  auto *ops = exec_config.slow_ops;
  const int raw_device_count =
      ops ? ops->getDeviceCount()
          : ::orteaf::internal::execution::mps::platform::wrapper::
                getDeviceCount();
  const std::size_t device_count = raw_device_count <= 0
                                         ? 0u
                                         : static_cast<std::size_t>(
                                               raw_device_count);

  auto &device_cfg = exec_config.device_config;
  if (device_cfg.payload_capacity == 0) {
    device_cfg.payload_capacity = device_count;
  }
  if (device_cfg.payload_block_size == 0 &&
      device_cfg.payload_capacity > 0) {
    device_cfg.payload_block_size = device_cfg.payload_capacity;
  }
  if (device_cfg.control_block_capacity == 0 &&
      device_cfg.payload_capacity > 0) {
    device_cfg.control_block_capacity = device_cfg.payload_capacity;
  }
  if (device_cfg.control_block_block_size == 0 &&
      device_cfg.control_block_capacity > 0) {
    device_cfg.control_block_block_size = device_cfg.control_block_capacity;
  }

  const std::size_t per_device_pool = device_count > 0 ? 1u : 0u;
  ensurePoolConfig(device_cfg.command_queue_config, per_device_pool);
  ensurePoolConfig(device_cfg.event_config, per_device_pool);
  ensurePoolConfig(device_cfg.fence_config, per_device_pool);
  ensurePoolConfig(device_cfg.heap_config, per_device_pool);
  ensurePoolConfig(device_cfg.heap_config.buffer_config, per_device_pool);
  ensurePoolConfig(device_cfg.library_config, per_device_pool);
  ensurePoolConfig(device_cfg.library_config.pipeline_config, per_device_pool);
  ensurePoolConfig(device_cfg.graph_config, per_device_pool);

  ensurePoolConfig(exec_config.kernel_base_config, 1u);

  auto &metadata_cfg = exec_config.kernel_metadata_config;
  const std::size_t metadata_capacity =
      metadata_cfg.payload_capacity > 0 ? metadata_cfg.payload_capacity : 1u;
  if (metadata_cfg.control_block_capacity == 0) {
    metadata_cfg.control_block_capacity = metadata_capacity;
  }
  if (metadata_cfg.control_block_block_size == 0) {
    metadata_cfg.control_block_block_size =
        metadata_cfg.control_block_capacity;
  }
  if (metadata_cfg.payload_block_size == 0) {
    metadata_cfg.payload_block_size = metadata_capacity;
  }
}
#endif

#if ORTEAF_ENABLE_CUDA
void applyCudaDefaults(LibraryConfig &config) {
  auto &exec_config = config.cuda_execution;
  auto *ops = exec_config.slow_ops;
  const int raw_device_count = [&]() {
    if (ops) {
      return ops->getDeviceCount();
    }
    ::orteaf::internal::execution::cuda::platform::wrapper::cudaInit();
    return ::orteaf::internal::execution::cuda::platform::wrapper::
        getDeviceCount();
  }();
  const std::size_t device_count = raw_device_count <= 0
                                         ? 0u
                                         : static_cast<std::size_t>(
                                               raw_device_count);

  auto &device_cfg = exec_config.device_config;
  if (device_cfg.payload_capacity == 0) {
    device_cfg.payload_capacity = device_count;
  }
  if (device_cfg.payload_block_size == 0 &&
      device_cfg.payload_capacity > 0) {
    device_cfg.payload_block_size = device_cfg.payload_capacity;
  }
  if (device_cfg.control_block_capacity == 0 &&
      device_cfg.payload_capacity > 0) {
    device_cfg.control_block_capacity = device_cfg.payload_capacity;
  }
  if (device_cfg.control_block_block_size == 0 &&
      device_cfg.control_block_capacity > 0) {
    device_cfg.control_block_block_size = device_cfg.control_block_capacity;
  }

  const std::size_t stream_capacity = device_count * 4u;
  auto &context_cfg = device_cfg.context_config;
  ensurePoolConfig(context_cfg, device_count);
  ensurePoolConfig(context_cfg.stream_config, stream_capacity);
  ensurePoolConfig(context_cfg.event_config, stream_capacity);
  ensurePoolConfig(context_cfg.buffer_config, stream_capacity);
  ensurePoolConfig(context_cfg.module_config, device_count);
}
#endif

}  // namespace

void initialize(const LibraryConfig &config) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Library is already initialized");
  }

  LibraryConfig resolved = config;

#if ORTEAF_ENABLE_MPS
  applyMpsDefaults(resolved);
#endif
#if ORTEAF_ENABLE_CUDA
  applyCudaDefaults(resolved);
#endif

  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::configure(
      resolved.cpu_execution);
#if ORTEAF_ENABLE_MPS
  ::orteaf::internal::execution::mps::api::MpsExecutionApi::configure(
      resolved.mps_execution);
#endif
#if ORTEAF_ENABLE_CUDA
  ::orteaf::internal::execution::cuda::api::CudaExecutionApi::configure(
      resolved.cuda_execution);
#endif
  ::orteaf::internal::tensor::api::TensorApi::configure(resolved.tensor_api);

  if (resolved.register_kernels) {
    ::orteaf::internal::kernel::registry::registerAllKernels();
  }

  g_initialized.store(true);
}

void shutdown() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_initialized) {
    return;
  }

  ::orteaf::internal::kernel::api::KernelRegistryApi::clear();
  ::orteaf::internal::tensor::api::TensorApi::shutdown();
  ::orteaf::internal::execution::cpu::api::CpuExecutionApi::shutdown();
  ::orteaf::internal::execution_context::cpu::reset();
#if ORTEAF_ENABLE_MPS
  ::orteaf::internal::execution::mps::api::MpsExecutionApi::shutdown();
  ::orteaf::internal::execution_context::mps::reset();
#endif
#if ORTEAF_ENABLE_CUDA
  ::orteaf::internal::execution::cuda::api::CudaExecutionApi::shutdown();
  ::orteaf::internal::execution_context::cuda::reset();
#endif

  g_initialized.store(false);
}

bool isInitialized() noexcept { return g_initialized.load(); }

}  // namespace orteaf::internal::init
