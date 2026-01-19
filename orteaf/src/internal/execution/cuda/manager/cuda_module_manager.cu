#include "orteaf/internal/execution/cuda/manager/cuda_module_manager.h"

#if ORTEAF_ENABLE_CUDA

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::cuda::manager {

bool ModulePayloadPoolTraits::create(Payload &payload, const Request &request,
                                     const Context &context) {
  if (context.ops == nullptr || context.context == nullptr) {
    return false;
  }
  if (request.key.identifier.empty()) {
    return false;
  }

  context.ops->setContext(context.context);

  ::orteaf::internal::execution::cuda::platform::wrapper::CudaModule_t cuda_module =
      nullptr;
  if (request.key.kind == ModuleKeyKind::kFile) {
    cuda_module = context.ops->loadModuleFromFile(request.key.identifier.c_str());
  } else {
    using namespace ::orteaf::internal::execution::cuda::platform::wrapper::
        kernel_embed;
    const auto preferred = request.key.preferred_format;
    Blob blob = findKernelData(request.key.identifier, preferred);
    if (blob.data == nullptr) {
      return false;
    }
    cuda_module = context.ops->loadModuleFromImage(blob.data);
  }

  if (cuda_module == nullptr) {
    return false;
  }

  payload.module = cuda_module;
  CudaFunctionManager::InternalConfig function_config{};
  function_config.public_config = context.function_config;
  function_config.context = context.context;
  function_config.module = cuda_module;
  function_config.ops = context.ops;
  payload.function_manager.configure(function_config);
  return true;
}

void ModulePayloadPoolTraits::destroy(Payload &payload, const Request &,
                                      const Context &context) {
  payload.function_manager.shutdown();
  if (payload.module != nullptr && context.ops != nullptr) {
    context.ops->unloadModule(payload.module);
  }
  payload.module = nullptr;
}

void CudaModuleManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.context == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA module manager requires a valid context");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "CUDA module manager requires valid ops");
  }

  context_ = config.context;
  ops_ = config.ops;
  const auto &cfg = config.public_config;
  function_config_ = cfg.function_config;

  std::size_t payload_capacity = cfg.payload_capacity;
  if (payload_capacity == 0) {
    payload_capacity = 4;
  }
  std::size_t payload_block_size = cfg.payload_block_size;
  if (payload_block_size == 0) {
    payload_block_size = 4;
  }
  std::size_t control_block_capacity = cfg.control_block_capacity;
  if (control_block_capacity == 0) {
    control_block_capacity = 4;
  }
  std::size_t control_block_block_size = cfg.control_block_block_size;
  if (control_block_block_size == 0) {
    control_block_block_size = 4;
  }

  key_to_index_.clear();

  const ModulePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<ModulePayloadPoolTraits::Request,
                ModulePayloadPoolTraits::Context>{}
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

void CudaModuleManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  lifetime_.clear();
  const ModulePayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);

  key_to_index_.clear();
  context_ = nullptr;
  ops_ = nullptr;
  function_config_ = {};
}

CudaModuleManager::ModuleLease CudaModuleManager::acquire(const ModuleKey &key) {
  core_.ensureConfigured();
  validateKey(key);

  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    const auto handle = ModuleHandle{
        static_cast<typename ModuleHandle::index_type>(it->second)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "CUDA module cache is invalid");
    }
    auto cached = lifetime_.get(handle);
    if (cached) {
      return cached;
    }
    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
  }

  ModulePayloadPoolTraits::Request request{key};
  const auto context = makePayloadContext();
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "CUDA module manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create CUDA module");
  }

  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

CudaModuleManager::ModuleLease CudaModuleManager::acquire(ModuleHandle handle) {
  core_.ensureConfigured();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid module handle");
  }
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  const auto *payload_ptr = lease.operator->();
  if (payload_ptr == nullptr || payload_ptr->module == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Module handle does not exist");
  }
  lifetime_.set(lease);
  return lease;
}

CudaModuleManager::FunctionType CudaModuleManager::getFunction(
    ModuleLease &lease, std::string_view name) {
  if (!lease) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Module lease is invalid");
  }
  if (name.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Kernel name cannot be empty");
  }
  auto *payload = lease.operator->();
  if (payload == nullptr || payload->module == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Module lease has no module loaded");
  }

  ops_->setContext(context_);
  return payload->function_manager.getFunction(std::string{name});
}

void CudaModuleManager::validateKey(const ModuleKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Module identifier cannot be empty");
  }
  if (key.kind == ModuleKeyKind::kEmbedded) {
    using namespace ::orteaf::internal::execution::cuda::platform::wrapper::
        kernel_embed;
    if (!available(key.identifier)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "Embedded module is not available");
    }
  }
}

ModulePayloadPoolTraits::Context
CudaModuleManager::makePayloadContext() const noexcept {
  ModulePayloadPoolTraits::Context context{};
  context.context = context_;
  context.ops = ops_;
  context.function_config = function_config_;
  return context;
}

} // namespace orteaf::internal::execution::cuda::manager

#endif // ORTEAF_ENABLE_CUDA
