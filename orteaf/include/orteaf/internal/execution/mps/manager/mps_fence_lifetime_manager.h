#pragma once

#if ORTEAF_ENABLE_MPS

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/manager/fifo_lease_lifetime_registry.h"
#include "orteaf/internal/diagnostics/error/error.h"
#include "orteaf/internal/diagnostics/log/log_config.h"
#include "orteaf/internal/execution/mps/manager/mps_fence_manager.h"
#include "orteaf/internal/execution/mps/mps_handles.h"
#include "orteaf/internal/execution/mps/platform/mps_fast_ops.h"
#include "orteaf/internal/execution/mps/resource/mps_fence_hazard.h"

namespace orteaf::internal::execution::mps::manager {

/**
 * @brief PayloadTraits for MpsFenceHazard used with FifoLeaseLifetimeRegistry.
 */
struct MpsFencePayloadTraits {
  using Payload = ::orteaf::internal::execution::mps::resource::MpsFenceHazard;

  template <typename FastOps> static bool isReady(Payload &payload) {
    return payload.isReady<FastOps>();
  }

  static void markCompletedUnsafe(Payload &payload) {
    payload.markCompletedUnsafe();
  }

  template <typename FastOps>
  static void validateBeforeRelease(Payload &payload) {
#if ORTEAF_MPS_DEBUG_ENABLED
    if (!payload.isReady<FastOps>()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence lifetime manager release ready aborted due to active "
          "fences");
    }
#else
    payload.markCompletedUnsafe();
#endif
  }

  template <typename FastOps> static void waitUntilReady(Payload &payload) {
    auto command_buffer = payload.commandBuffer();
    if (command_buffer != nullptr) {
      FastOps::waitUntilCompleted(command_buffer);
    }
  }
};

/**
 * @brief MPS fence lifetime manager.
 *
 * Manages fence leases in FIFO order for a specific command queue.
 * Wraps FifoLeaseLifetimeRegistry with MPS-specific acquire logic.
 */
class MpsFenceLifetimeManager {
public:
  using FenceManager =
      ::orteaf::internal::execution::mps::manager::MpsFenceManager;
  using StrongFenceLease = FenceManager::StrongFenceLease;
  using CommandQueueHandle =
      ::orteaf::internal::execution::mps::MpsCommandQueueHandle;
  using CommandBufferType =
      ::orteaf::internal::execution::mps::platform::wrapper::MpsCommandBuffer_t;

  using Registry = ::orteaf::internal::base::manager::FifoLeaseLifetimeRegistry<
      StrongFenceLease, MpsFencePayloadTraits>;

  MpsFenceLifetimeManager() = default;
  MpsFenceLifetimeManager(const MpsFenceLifetimeManager &) = delete;
  MpsFenceLifetimeManager &operator=(const MpsFenceLifetimeManager &) = delete;
  MpsFenceLifetimeManager(MpsFenceLifetimeManager &&) noexcept = default;
  MpsFenceLifetimeManager &
  operator=(MpsFenceLifetimeManager &&) noexcept = default;
  ~MpsFenceLifetimeManager() = default;

  bool setFenceManager(FenceManager *manager) noexcept {
    if (!empty() && manager != fence_manager_) {
      return false;
    }
    fence_manager_ = manager;
    return true;
  }

  bool setCommandQueueHandle(CommandQueueHandle handle) noexcept {
    if (!empty() && handle != queue_handle_) {
      return false;
    }
    queue_handle_ = handle;
    return true;
  }

  StrongFenceLease acquire(CommandBufferType command_buffer) {
    if (fence_manager_ == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence lifetime manager requires a fence manager");
    }
    if (!queue_handle_.isValid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "MPS fence lifetime manager requires a valid command queue handle");
    }
    if (command_buffer == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "MPS fence lifetime manager requires a command buffer");
    }
    auto lease = fence_manager_->acquire();
    auto *payload = lease.operator->();
    if (payload == nullptr) {
      lease.release();
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence lease has no payload");
    }
    if (!payload->setCommandQueueHandle(queue_handle_)) {
      lease.release();
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence hazard failed to bind command queue handle");
    }
    if (!payload->setCommandBuffer(command_buffer)) {
      lease.release();
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence hazard failed to bind command buffer");
    }
    registry_.push(lease);
    return lease;
  }

  template <typename FastOps =
                ::orteaf::internal::execution::mps::platform::MpsFastOps>
  std::size_t releaseReady() {
    ensureConfigured();
    return registry_.releaseReady<FastOps>();
  }

  template <typename FastOps =
                ::orteaf::internal::execution::mps::platform::MpsFastOps>
  void clear() {
    ensureConfigured();
    registry_.clear<FastOps>();
  }

  template <typename FastOps =
                ::orteaf::internal::execution::mps::platform::MpsFastOps>
  std::size_t waitUntilReady() {
    ensureConfigured();
    return registry_.waitUntilReady<FastOps>();
  }

  std::size_t size() const noexcept { return registry_.size(); }

  bool empty() const noexcept { return registry_.empty(); }

#if ORTEAF_ENABLE_TEST
  std::size_t storageSizeForTest() const noexcept {
    return registry_.storageSizeForTest();
  }
  std::size_t headIndexForTest() const noexcept {
    return registry_.headIndexForTest();
  }
#endif

private:
  void ensureConfigured() const {
    if (fence_manager_ == nullptr) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
          "MPS fence lifetime manager requires a fence manager");
    }
    if (!queue_handle_.isValid()) {
      ::orteaf::internal::diagnostics::error::throwError(
          ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidArgument,
          "MPS fence lifetime manager requires a valid command queue handle");
    }
  }

  FenceManager *fence_manager_{nullptr};
  CommandQueueHandle queue_handle_{CommandQueueHandle::invalid()};
  Registry registry_{};
};

} // namespace orteaf::internal::execution::mps::manager

#endif // ORTEAF_ENABLE_MPS
