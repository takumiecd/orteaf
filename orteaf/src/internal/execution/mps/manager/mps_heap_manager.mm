#include "orteaf/internal/execution/mps/manager/mps_heap_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

namespace {

constexpr std::uint64_t kMtlResourceStorageModeShift = 4u;
constexpr std::uint64_t kMtlResourceStorageModeMask = 0xF0u;

inline ::orteaf::internal::execution::mps::platform::wrapper::MpsBufferUsage_t
composeUsageWithStorageMode(
    std::uint64_t resource_options,
    std::uint64_t storage_mode) {
  const auto cleared = resource_options & ~kMtlResourceStorageModeMask;
  const auto mode_bits =
      (storage_mode << kMtlResourceStorageModeShift) &
      kMtlResourceStorageModeMask;
  return static_cast<::orteaf::internal::execution::mps::platform::wrapper::
                         MpsBufferUsage_t>(cleared | mode_bits);
}

} // namespace

// =============================================================================
// HeapPayloadPoolTraits Implementation
// =============================================================================

bool MpsHeapPayload::initialize(const InitConfig &config) {
  if (config.ops == nullptr || config.device == nullptr) {
    return false;
  }

  // Create heap descriptor
  auto descriptor = config.ops->createHeapDescriptor();
  if (descriptor == nullptr) {
    return false;
  }

  // RAII guard for descriptor cleanup
  struct DescriptorGuard {
    ::orteaf::internal::execution::mps::platform::wrapper::MpsHeapDescriptor_t
        handle{nullptr};
    SlowOps *slow_ops{nullptr};
    ~DescriptorGuard() {
      if (handle != nullptr && slow_ops != nullptr) {
        slow_ops->destroyHeapDescriptor(handle);
      }
    }
  };
  DescriptorGuard guard{descriptor, config.ops};

  // Configure descriptor
  config.ops->setHeapDescriptorSize(descriptor, config.key.size_bytes);
  config.ops->setHeapDescriptorResourceOptions(descriptor,
                                               config.key.resource_options);
  config.ops->setHeapDescriptorStorageMode(descriptor,
                                           config.key.storage_mode);
  config.ops->setHeapDescriptorCPUCacheMode(descriptor,
                                            config.key.cpu_cache_mode);
  config.ops->setHeapDescriptorHazardTrackingMode(
      descriptor, config.key.hazard_tracking_mode);
  config.ops->setHeapDescriptorType(descriptor, config.key.heap_type);

  // Create heap
  heap_ = config.ops->createHeap(config.device, descriptor);
  if (heap_ == nullptr) {
    return false;
  }

  // Configure buffer manager
  BufferManager::InternalConfig buf_cfg{};
  buf_cfg.public_config = config.buffer_config;
  buf_cfg.usage = composeUsageWithStorageMode(
      static_cast<std::uint64_t>(config.key.resource_options),
      static_cast<std::uint64_t>(config.key.storage_mode));
  buf_cfg.device = config.device;
  buf_cfg.device_handle = config.device_handle;
  buf_cfg.heap = heap_;
  buf_cfg.library_manager = config.library_manager;
  buffer_manager_.configure(buf_cfg);

  return true;
}

bool HeapPayloadPoolTraits::create(Payload &payload, const Request &request,
                                   const Context &context) {
  MpsHeapPayload::InitConfig init{};
  init.device = context.device;
  init.device_handle = context.device_handle;
  init.library_manager = context.library_manager;
  init.ops = context.ops;
  init.key = request.key;
  init.buffer_config = context.buffer_config;
  return payload.initialize(init);
}

void HeapPayloadPoolTraits::destroy(Payload &payload, const Request &,
                                    const Context &context) {
  payload.reset(context.ops);
}

// =============================================================================
// MpsHeapManager Implementation
// =============================================================================

void MpsHeapManager::configure(const InternalConfig &config) {
  shutdown();

  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS heap manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS heap manager requires valid ops");
  }

  device_ = config.device;
  device_handle_ = config.device_handle;
  library_manager_ = config.library_manager;
  ops_ = config.ops;
  const auto &cfg = config.public_config;
  buffer_config_ = cfg.buffer_config;
  key_to_index_.clear();

  // Configure core + payload pool
  HeapPayloadPoolTraits::Request request{};
  auto context = makePayloadContext();
  Core::Builder<HeapPayloadPoolTraits::Request,
                HeapPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(
          cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(request)
      .withContext(context)
      .configure(core_);

}

void MpsHeapManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  lifetime_.clear();

  // Destroy all payloads
  auto context = makePayloadContext();
  HeapPayloadPoolTraits::Request request{};
  core_.shutdown(request, context);

  // Shutdown control block pool

  key_to_index_.clear();
  device_ = nullptr;
  device_handle_ = {};
  library_manager_ = nullptr;
  ops_ = nullptr;
}

MpsHeapManager::HeapLease
MpsHeapManager::acquire(const HeapDescriptorKey &key) {
  core_.ensureConfigured();
  validateKey(key);

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    const auto index = it->second;
    const HeapHandle handle{
        static_cast<typename HeapHandle::index_type>(index)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "Cached heap index is invalid");
    }
    auto cached = lifetime_.get(handle);
    if (cached) {
      return cached;
    }
    auto lease = core_.acquireStrongLease(handle);
    lifetime_.set(lease);
    return lease;
  }

  // Reserve an uncreated slot and create the heap
  HeapPayloadPoolTraits::Request request{key};
  auto context = makePayloadContext();
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "Heap manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create heap");
  }

  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

MpsHeapManager::HeapLease
MpsHeapManager::acquire(HeapHandle handle) {
  core_.ensureConfigured();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Heap handle is invalid");
  }
  if (!core_.isAlive(handle)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Heap handle does not reference a live payload");
  }
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

void MpsHeapManager::validateKey(const HeapDescriptorKey &key) const {
  if (key.size_bytes == 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Heap size must be > 0");
  }
}

HeapPayloadPoolTraits::Context
MpsHeapManager::makePayloadContext() const noexcept {
  HeapPayloadPoolTraits::Context ctx{};
  ctx.device = device_;
  ctx.device_handle = device_handle_;
  ctx.library_manager = library_manager_;
  ctx.ops = ops_;
  ctx.buffer_config = buffer_config_;
  return ctx;
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
