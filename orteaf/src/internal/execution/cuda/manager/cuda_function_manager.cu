#include "orteaf/internal/execution/cuda/manager/cuda_function_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

void CudaFunctionManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA function manager requires a valid context");
  }
  if (config.module == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA function manager requires a valid module");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA function manager requires valid ops");
  }

  context_ = config.context;
  module_ = config.module;
  ops_ = config.ops;
  const auto &cfg = config.public_config;

  std::size_t payload_capacity = cfg.payload_capacity;
  if (payload_capacity == 0) {
    payload_capacity = 8;
  }
  std::size_t payload_block_size = cfg.payload_block_size;
  if (payload_block_size == 0) {
    payload_block_size = 8;
  }
  std::size_t control_block_capacity = cfg.control_block_capacity;
  if (control_block_capacity == 0) {
    control_block_capacity = 8;
  }
  std::size_t control_block_block_size = cfg.control_block_block_size;
  if (control_block_block_size == 0) {
    control_block_block_size = 8;
  }

  name_to_index_.clear();

  const FunctionPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<FunctionPayloadPoolTraits::Request,
                FunctionPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(control_block_capacity)
      .withControlBlockBlockSize(control_block_block_size)
      .withControlBlockGrowthChunkSize(cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(payload_capacity)
      .withPayloadBlockSize(payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext(payload_context)
      .configure(core_);
}

void CudaFunctionManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  lifetime_.clear();
  const FunctionPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);

  name_to_index_.clear();
  context_ = nullptr;
  module_ = nullptr;
  ops_ = nullptr;
}

CudaFunctionManager::FunctionLease
CudaFunctionManager::acquire(std::string_view name) {
  core_.ensureConfigured();
  validateName(name);

  const std::string key{name};
  if (auto it = name_to_index_.find(key); it != name_to_index_.end()) {
    const auto handle = FunctionHandle{
        static_cast<typename FunctionHandle::index_type>(it->second)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA function cache is invalid");
    }
    auto cached = lifetime_.get(handle);
    if (cached) {
      return cached;
    }
    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
  }

  FunctionPayloadPoolTraits::Request request{};
  request.name = key;
  const auto context = makePayloadContext();
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA function manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create CUDA function");
  }

  name_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

CudaFunctionManager::FunctionLease
CudaFunctionManager::acquire(FunctionHandle handle) {
  core_.ensureConfigured();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid function handle");
  }
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  const auto *payload_ptr = lease.operator->();
  if (payload_ptr == nullptr || *payload_ptr == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Function handle does not exist");
  }
  lifetime_.set(lease);
  return lease;
}

CudaFunctionManager::FunctionType
CudaFunctionManager::getFunction(std::string_view name) {
  auto lease = acquire(name);
  auto *payload_ptr = lease.operator->();
  if (payload_ptr == nullptr || *payload_ptr == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to resolve CUDA function");
  }
  return *payload_ptr;
}

void CudaFunctionManager::validateName(std::string_view name) const {
  if (name.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Function name cannot be empty");
  }
}

FunctionPayloadPoolTraits::Context
CudaFunctionManager::makePayloadContext() const noexcept {
  FunctionPayloadPoolTraits::Context context{};
  context.context = context_;
  context.module = module_;
  context.ops = ops_;
  return context;
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
