#include "orteaf/internal/execution/mps/manager/mps_library_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsLibraryManager::configure(const Config &config) {
  shutdown();
  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS library manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS library manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  pipeline_config_ = config.pipeline_config;
  // payload block size managed by core_
  // payload growth chunk size configured via core_
  key_to_index_.clear();

  const LibraryPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.configure(config.pool, payload_request, payload_context);
  core_.setConfigured(true);
}

void MpsLibraryManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }

  core_.checkCanShutdownOrThrow();
  const LibraryPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdownPayloadPool(payload_request, payload_context);
  core_.shutdownControlBlockPool();

  key_to_index_.clear();
  device_ = nullptr;
  ops_ = nullptr;
}

MpsLibraryManager::LibraryLease
MpsLibraryManager::acquire(const LibraryKey &key) {
  core_.ensureConfigured();
  validateKey(key);

  // Check cache first
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    const auto handle = LibraryHandle{
        static_cast<typename LibraryHandle::index_type>(it->second)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS library cache is invalid");
    }
    return core_.acquireWeakLease(handle);
  }

  // Reserve an uncreated slot and create the library
  LibraryPayloadPoolTraits::Request request{key};
  const auto context = makePayloadContext();
  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS library manager has no available slots");
  }
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS library");
  }

  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  return core_.acquireWeakLease(handle);
}

MpsLibraryManager::LibraryLease
MpsLibraryManager::acquire(LibraryHandle handle) {
  core_.ensureConfigured();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid library handle");
  }
  auto lease = core_.acquireWeakLease(handle);
  const auto *payload_ptr = lease.payloadPtr();
  if (payload_ptr == nullptr || payload_ptr->library == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Library handle does not exist");
  }
  return lease;
}

void MpsLibraryManager::validateKey(const LibraryKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Library identifier cannot be empty");
  }
}

LibraryPayloadPoolTraits::Context
MpsLibraryManager::makePayloadContext() const noexcept {
  LibraryPayloadPoolTraits::Context context{};
  context.device = device_;
  context.ops = ops_;
  context.pipeline_config = pipeline_config_;
  return context;
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
