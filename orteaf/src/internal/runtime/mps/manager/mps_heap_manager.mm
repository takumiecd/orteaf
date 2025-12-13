#include "orteaf/internal/runtime/mps/manager/mps_heap_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::runtime::mps::manager {

void MpsHeapManager::initialize(DeviceType device, SlowOps *ops,
                                std::size_t capacity) {
  shutdown();
  if (device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS heap manager requires a valid device");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS heap manager requires valid ops");
  }
  if (capacity > static_cast<std::size_t>(HeapHandle::invalid_index())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS heap manager capacity exceeds maximum handle range");
  }
  device_ = device;
  ops_ = ops;
  clearCacheStates();
  key_to_index_.clear();
  if (capacity > 0) {
    states_.reserve(capacity);
  }
  initialized_ = true;
}

void MpsHeapManager::shutdown() {
  if (!initialized_) {
    return;
  }
  for (std::size_t i = 0; i < states_.size(); ++i) {
    State &state = states_[i];
    if (state.alive && state.resource.heap != nullptr) {
      ops_->destroyHeap(state.resource.heap);
      state.resource.heap = nullptr;
      state.alive = false;
    }
  }
  clearCacheStates();
  key_to_index_.clear();
  device_ = nullptr;
  ops_ = nullptr;
  initialized_ = false;
}

MpsHeapManager::HeapLease
MpsHeapManager::acquire(const HeapDescriptorKey &key) {
  ensureInitialized();
  validateKey(key);

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    incrementUseCount(it->second);
    return HeapLease{this, createHandle<HeapHandle>(it->second),
                     states_[it->second].resource.heap};
  }

  // Create new entry
  const std::size_t index = allocateSlot();
  State &state = states_[index];
  state.resource.heap = createHeap(key);
  markSlotAlive(index);
  key_to_index_.emplace(key, index);

  return HeapLease{this, createHandle<HeapHandle>(index), state.resource.heap};
}

void MpsHeapManager::release(HeapLease &lease) noexcept {
  if (!lease) {
    return;
  }
  decrementUseCount(static_cast<std::size_t>(lease.handle().index));
  lease.invalidate();
}

void MpsHeapManager::validateKey(const HeapDescriptorKey &key) const {
  if (key.size_bytes == 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Heap size must be > 0");
  }
}

::orteaf::internal::runtime::mps::platform::wrapper::MPSHeap_t
MpsHeapManager::createHeap(const HeapDescriptorKey &key) {
  auto descriptor = ops_->createHeapDescriptor();
  if (descriptor == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to allocate MPS heap descriptor");
  }
  struct DescriptorGuard {
    ::orteaf::internal::runtime::mps::platform::wrapper::MPSHeapDescriptor_t
        handle{nullptr};
    SlowOps *slow_ops{nullptr};
    ~DescriptorGuard() {
      if (handle != nullptr && slow_ops != nullptr) {
        slow_ops->destroyHeapDescriptor(handle);
      }
    }
  };
  DescriptorGuard guard{descriptor, ops_};
  ops_->setHeapDescriptorSize(descriptor, key.size_bytes);
  ops_->setHeapDescriptorResourceOptions(descriptor, key.resource_options);
  ops_->setHeapDescriptorStorageMode(descriptor, key.storage_mode);
  ops_->setHeapDescriptorCPUCacheMode(descriptor, key.cpu_cache_mode);
  ops_->setHeapDescriptorHazardTrackingMode(descriptor,
                                            key.hazard_tracking_mode);
  ops_->setHeapDescriptorType(descriptor, key.heap_type);
  auto heap = ops_->createHeap(device_, descriptor);
  ops_->destroyHeapDescriptor(descriptor);
  guard.handle = nullptr;
  if (heap == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS heap");
  }
  return heap;
}

} // namespace orteaf::internal::runtime::mps::manager

#endif // ORTEAF_ENABLE_MPS
