#include "orteaf/internal/runtime/manager/mps/mps_library_manager.h"

#if ORTEAF_ENABLE_MPS

namespace orteaf::internal::runtime::mps {

void MpsLibraryManager::initialize(
    ::orteaf::internal::backend::mps::MPSDevice_t device, BackendOps *ops,
    std::size_t capacity) {
  shutdown();
  if (device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS library manager requires a valid device");
  }
  if (ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS library manager requires valid ops");
  }
  device_ = device;
  ops_ = ops;
  if (capacity > base::LibraryHandle::invalid_index()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Requested MPS library capacity exceeds supported limit");
  }
  states_.clear();
  free_list_.clear();
  key_to_index_.clear();
  states_.reserve(capacity);
  free_list_.reserve(capacity);
  for (std::size_t i = 0; i < capacity; ++i) {
    states_.pushBack(State{});
    free_list_.pushBack(i);
  }
  initialized_ = true;
}

void MpsLibraryManager::shutdown() {
  if (!initialized_) {
    return;
  }
  for (std::size_t i = 0; i < states_.size(); ++i) {
    State &state = states_[i];
    if (state.alive) {
      state.pipeline_manager.shutdown();
      ops_->destroyLibrary(state.handle);
    }
  }
  states_.clear();
  free_list_.clear();
  key_to_index_.clear();
  device_ = nullptr;
  ops_ = nullptr;
  initialized_ = false;
}

MpsLibraryManager::LibraryLease MpsLibraryManager::acquire(const LibraryKey &key) {
  ensureInitialized();
  validateKey(key);
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    return acquireLibraryFromHandle(encodeHandle(it->second));
  }
  const std::size_t index = allocateSlot();
  State &state = states_[index];
  state.handle = createLibrary(key);
  state.key = key;
  state.pipeline_manager.initialize(device_, state.handle, ops_, 0);
  state.alive = true;
  const auto handle = encodeHandle(index);
  key_to_index_.emplace(state.key, index);
  return LibraryLease{this, handle, state.handle};
}

MpsLibraryManager::LibraryLease MpsLibraryManager::acquire(const PipelineManagerLease &pipeline_lease) {
  return acquireLibraryFromHandle(pipeline_lease.handle());
}

MpsLibraryManager::PipelineManagerLease
MpsLibraryManager::acquirePipelineManager(const LibraryLease &lease) {
  State &state = ensureAliveState(lease.handle());
  return PipelineManagerLease{this, lease.handle(), &state.pipeline_manager};
}

MpsLibraryManager::PipelineManagerLease
MpsLibraryManager::acquirePipelineManager(const LibraryKey &key) {
  ensureInitialized();
  validateKey(key);

  std::size_t index = 0;
  State *state = nullptr;
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    index = it->second;
    state = &states_[index];
    ensureAliveState(encodeHandle(index));
  } else {
    index = allocateSlot();
    state = &states_[index];
    state->handle = createLibrary(key);
    state->key = key;
    state->pipeline_manager.initialize(device_, state->handle, ops_, 0);
    state->alive = true;
    key_to_index_.emplace(state->key, index);
  }

  const auto handle = encodeHandle(index);
  return PipelineManagerLease{this, handle, &state->pipeline_manager};
}

MpsLibraryManager::LibraryLease MpsLibraryManager::acquireLibraryFromHandle(base::LibraryHandle handle) {
  State &state = ensureAliveState(handle);
  return LibraryLease{this, handle, state.handle};
}

void MpsLibraryManager::release(LibraryLease &lease) noexcept {
  if (!lease) {
    return;
  }
  releaseHandle(lease.handle());
  lease.invalidate();
}

void MpsLibraryManager::release(PipelineManagerLease &lease) noexcept {
  if (!lease) {
    return;
  }
  releaseHandle(lease.handle());
  lease.invalidate();
}

void MpsLibraryManager::releaseHandle(base::LibraryHandle handle) noexcept {
  // No-op: Libraries are not released until shutdown
}

#if ORTEAF_ENABLE_TEST
MpsLibraryManager::DebugState
MpsLibraryManager::debugState(base::LibraryHandle handle) const {
  DebugState snapshot{};
  snapshot.growth_chunk_size = growth_chunk_size_;
  const std::size_t index = static_cast<std::size_t>(handle.index);
  if (index < states_.size()) {
    const State &state = states_[index];
    snapshot.alive = state.alive;
    snapshot.handle_allocated = state.handle != nullptr;
    snapshot.kind = state.key.kind;
    snapshot.identifier = state.key.identifier;
  }
  return snapshot;
}
#endif

void MpsLibraryManager::ensureInitialized() const {
  if (!initialized_ || device_ == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS library manager not initialized");
  }
}

void MpsLibraryManager::validateKey(const LibraryKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Library identifier cannot be empty");
  }
}

MpsLibraryManager::State &
MpsLibraryManager::ensureAliveState(base::LibraryHandle handle) {
  ensureInitialized();
  const std::size_t index = static_cast<std::size_t>(handle.index);
  if (index >= states_.size()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS library handle out of range");
  }
  State &state = states_[index];
  if (!state.alive) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "MPS library handle is inactive");
  }
  return state;
}

std::size_t MpsLibraryManager::allocateSlot() {
  if (free_list_.empty()) {
    growStatePool(growth_chunk_size_);
    if (free_list_.empty()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "No available MPS library slots");
    }
  }
  const std::size_t index = free_list_.back();
  free_list_.resize(free_list_.size() - 1);
  return index;
}

void MpsLibraryManager::growStatePool(std::size_t additional) {
  if (additional == 0) {
    return;
  }
  if (additional > (base::LibraryHandle::invalid_index() - states_.size())) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Requested MPS library capacity exceeds supported limit");
  }
  const std::size_t start = states_.size();
  states_.reserve(states_.size() + additional);
  free_list_.reserve(free_list_.size() + additional);
  for (std::size_t offset = 0; offset < additional; ++offset) {
    states_.pushBack(State{});
    free_list_.pushBack(start + offset);
  }
}

base::LibraryHandle MpsLibraryManager::encodeHandle(std::size_t index) const {
  return base::LibraryHandle{static_cast<std::uint32_t>(index)};
}

::orteaf::internal::backend::mps::MPSLibrary_t
MpsLibraryManager::createLibrary(const LibraryKey &key) {
  switch (key.kind) {
  case LibraryKeyKind::kNamed:
    return ops_->createLibraryWithName(device_, key.identifier);
  }
  ::orteaf::internal::diagnostics::error::throwError(
      ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
      "Unsupported MPS library key kind");
}

} // namespace orteaf::internal::runtime::mps

#endif // ORTEAF_ENABLE_MPS
