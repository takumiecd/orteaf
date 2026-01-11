#include "orteaf/internal/execution/mps/manager/mps_graph_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsGraphManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS graph manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS graph manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  const auto &cfg = config.public_config;
  key_to_index_.clear();
  const GraphPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<GraphPayloadPoolTraits::Request,
                GraphPayloadPoolTraits::Context>{}
      .withControlBlockCapacity(cfg.control_block_capacity)
      .withControlBlockBlockSize(cfg.control_block_block_size)
      .withControlBlockGrowthChunkSize(
          cfg.control_block_growth_chunk_size)
      .withPayloadCapacity(cfg.payload_capacity)
      .withPayloadBlockSize(cfg.payload_block_size)
      .withPayloadGrowthChunkSize(cfg.payload_growth_chunk_size)
      .withRequest(payload_request)
      .withContext(payload_context)
      .configure(core_);
}

void MpsGraphManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  const GraphPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);
  key_to_index_.clear();
  device_ = nullptr;
  ops_ = nullptr;
}

MpsGraphManager::GraphLease
MpsGraphManager::acquire(const GraphKey &key, const CompileFn &compile_fn) {
  core_.ensureConfigured();
  validateKey(key);
  if (!compile_fn) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS graph compile function cannot be empty");
  }

  // Check if already cached
  if (auto it = key_to_index_.find(key); it != key_to_index_.end()) {
    const auto handle =
        GraphHandle{static_cast<typename GraphHandle::index_type>(it->second)};
    if (!core_.isAlive(handle)) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS graph cache is invalid");
    }
    return core_.acquireStrongLease(handle);
  }

  auto handle = core_.reserveUncreatedPayloadOrGrow();
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS graph manager has no available slots");
  }

  GraphPayloadPoolTraits::Request request{&compile_fn};
  const auto context = makePayloadContext();
  if (!core_.emplacePayload(handle, request, context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS graph executable");
  }

  key_to_index_.emplace(key, static_cast<std::size_t>(handle.index));
  return core_.acquireStrongLease(handle);
}

void MpsGraphManager::validateKey(const GraphKey &key) const {
  if (key.identifier.empty()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Graph identifier cannot be empty");
  }
  if (key.data_type == ::orteaf::internal::execution::mps::platform::wrapper::
                           MpsGraphDataType::kInvalid) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Graph data type must be valid");
  }
  if (key.target_tensor_count == 0) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Graph target tensor count must be > 0");
  }
}

GraphPayloadPoolTraits::Context
MpsGraphManager::makePayloadContext() const noexcept {
  return GraphPayloadPoolTraits::Context{device_, ops_};
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
