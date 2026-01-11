#include "orteaf/internal/execution/mps/manager/mps_event_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsEventManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS event manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS event manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  const auto &cfg = config.public_config;

  const EventPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  Core::Builder<EventPayloadPoolTraits::Request,
                EventPayloadPoolTraits::Context>{}
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
  if (!core_.createAllPayloads(payload_request, payload_context)) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
        "Failed to create MPS events");
  }
}

void MpsEventManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  // Check canShutdown on all created control blocks

  const EventPayloadPoolTraits::Request payload_request{};
  const auto payload_context = makePayloadContext();
  core_.shutdown(payload_request, payload_context);

  device_ = nullptr;
  ops_ = nullptr;
}

MpsEventManager::EventLease MpsEventManager::acquire() {
  core_.ensureConfigured();
  const EventPayloadPoolTraits::Request request{};
  const auto context = makePayloadContext();
  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS event manager has no available slots");
  }
  return core_.acquireStrongLease(handle);
}

EventPayloadPoolTraits::Context
MpsEventManager::makePayloadContext() const noexcept {
  return EventPayloadPoolTraits::Context{device_, ops_};
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
