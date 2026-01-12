#include "orteaf/internal/execution/mps/manager/mps_heap_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

// =============================================================================
// HeapPayloadPoolTraits Implementation
// =============================================================================

bool HeapPayloadPoolTraits::create(Payload &payload, const Request &request,
                                   const Context &context) {
  if (context.ops == nullptr || context.device == nullptr) {
    return false;
  }

  // Create heap descriptor
  auto descriptor = context.ops->createHeapDescriptor();
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
  DescriptorGuard guard{descriptor, context.ops};

  // Configure descriptor
  context.ops->setHeapDescriptorSize(descriptor, request.key.size_bytes);
  context.ops->setHeapDescriptorResourceOptions(descriptor,
                                                request.key.resource_options);
  context.ops->setHeapDescriptorStorageMode(descriptor,
                                            request.key.storage_mode);
  context.ops->setHeapDescriptorCPUCacheMode(descriptor,
                                             request.key.cpu_cache_mode);
  context.ops->setHeapDescriptorHazardTrackingMode(
      descriptor, request.key.hazard_tracking_mode);
  context.ops->setHeapDescriptorType(descriptor, request.key.heap_type);

  // Create heap
  payload.heap = context.ops->createHeap(context.device, descriptor);
  if (payload.heap == nullptr) {
    return false;
  }

  // Configure buffer manager
  BufferManager::InternalConfig buf_cfg{};
  buf_cfg.public_config = context.buffer_config;
  buf_cfg.device = context.device;
  buf_cfg.device_handle = context.device_handle;
  buf_cfg.heap = payload.heap;
  buf_cfg.library_manager = context.library_manager;
  payload.buffer_manager.configure(buf_cfg);

  return true;
}

void HeapPayloadPoolTraits::destroy(Payload &payload, const Request &,
                                    const Context &context) {
  // Shutdown buffer manager first
  payload.buffer_manager.shutdown();

  // Then destroy heap
  if (payload.heap != nullptr && context.ops != nullptr) {
    context.ops->destroyHeap(payload.heap);
    payload.heap = nullptr;
  }
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

MpsHeapManager::BufferManager *
MpsHeapManager::bufferManager(const HeapLease &lease) {
  if (!lease) {
    return nullptr;
  }
  auto *payload = const_cast<MpsHeapResource *>(lease.operator->());
  return payload ? &payload->buffer_manager : nullptr;
}

MpsHeapManager::BufferManager *
MpsHeapManager::bufferManager(const HeapDescriptorKey &key) {
  core_.ensureConfigured();
  validateKey(key);

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    const auto index = it->second;
    const HeapHandle handle{
        static_cast<typename HeapHandle::index_type>(index)};
    auto lease = lifetime_.get(handle);
    if (!lease) {
      lease = core_.acquireStrongLease(handle);
      lifetime_.set(lease);
    }
    auto *payload = lease.operator->();
    return payload ? &payload->buffer_manager : nullptr;
  }

  // Create new entry - use acquire and then extract buffer_manager
  auto lease = acquire(key);
  if (!lease) {
    return nullptr;
  }
  auto *payload = lease.operator->();
  return payload ? &payload->buffer_manager : nullptr;
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
