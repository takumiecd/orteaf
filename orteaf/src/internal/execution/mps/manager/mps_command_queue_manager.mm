#include "orteaf/internal/execution/mps/manager/mps_command_queue_manager.h"

#if ORTEAF_ENABLE_MPS

#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::execution::mps::manager {

void MpsCommandQueueManager::configure(const InternalConfig &config) {
  shutdown();
  if (config.device == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires a valid device");
  }
  if (config.ops == nullptr) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "MPS command queue manager requires valid ops");
  }
  device_ = config.device;
  ops_ = config.ops;
  fence_manager_ = config.fence_manager;
  const auto &cfg = config.public_config;
  // payload block size managed by core_
  // payload growth chunk size configured via core_

  const CommandQueuePayloadPoolTraits::Request payload_request{};
  const CommandQueuePayloadPoolTraits::Context payload_context{device_, ops_,
                                                               fence_manager_};
  Core::Builder<CommandQueuePayloadPoolTraits::Request,
                CommandQueuePayloadPoolTraits::Context>{}
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
        "Failed to create MPS command queues");
  }
}

void MpsCommandQueueManager::shutdown() {
  if (!core_.isConfigured()) {
    return;
  }
  // Check canShutdown on all created control blocks
  lifetime_.clear();

  const CommandQueuePayloadPoolTraits::Request payload_request{};
  const CommandQueuePayloadPoolTraits::Context payload_context{device_, ops_,
                                                               fence_manager_};
  core_.shutdown(payload_request, payload_context);

  device_ = nullptr;
  ops_ = nullptr;
  fence_manager_ = nullptr;
}

MpsCommandQueueManager::CommandQueueLease MpsCommandQueueManager::acquire() {
  core_.ensureConfigured();
  const CommandQueuePayloadPoolTraits::Request request{};
  const CommandQueuePayloadPoolTraits::Context context{device_, ops_,
                                                       fence_manager_};

  auto handle = core_.acquirePayloadOrGrowAndCreate(request, context);
  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::OutOfRange,
        "MPS command queue manager has no available slots");
  }
  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  lifetime_.set(lease);
  return lease;
}

MpsCommandQueueManager::CommandQueueLease
MpsCommandQueueManager::acquire(CommandQueueHandle handle) {
  core_.ensureConfigured();

  if (!handle.isValid()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Invalid command queue handle");
  }

  auto cached = lifetime_.get(handle);
  if (cached) {
    return cached;
  }
  auto lease = core_.acquireStrongLease(handle);
  const auto *payload_ptr = lease.payloadPtr();
  if (payload_ptr == nullptr || !payload_ptr->hasQueue()) {
    ::orteaf::internal::diagnostics::error::throwError(
        ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
        "Command queue handle does not exist");
  }
  lifetime_.set(lease);
  return lease;
}

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
