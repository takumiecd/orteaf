#pragma once

#include <cstddef>
#include <utility>

#include "orteaf/internal/base/heap_vector.h"
#include "orteaf/internal/diagnostics/error/error.h"

namespace orteaf::internal::base::manager {

/**
 * @brief PayloadTraits concept for FifoLeaseLifetimeRegistry.
 *
 * PayloadTraits must provide:
 * - `using Payload = ...;` - The payload type
 * - `template <typename FastOps> static bool isReady(Payload&)` -
 *     Check if payload is ready for release
 * - `static void markCompletedUnsafe(Payload&)` - Mark payload as completed
 *     (only called in non-debug mode after FIFO order validation)
 * - `template <typename FastOps> static void validateBeforeRelease(Payload&)` -
 *     Optional debug validation before release (can throw)
 */
template <typename PayloadTraitsT>
concept FifoLeasePayloadTraits =
    requires(typename PayloadTraitsT::Payload &payload,
             const typename PayloadTraitsT::Payload &const_payload) {
      typename PayloadTraitsT::Payload;
    };

/**
 * @brief FIFO-ordered lease lifetime registry.
 *
 * Manages leases in FIFO order, allowing batch release of completed leases
 * from the front of the queue. This is useful for GPU fence management where
 * leases must be released in order of submission.
 *
 * @tparam StrongLeaseT The strong lease type
 * @tparam PayloadTraitsT Traits providing isReady/markCompletedUnsafe
 */
template <typename StrongLeaseT, typename PayloadTraitsT>
  requires FifoLeasePayloadTraits<PayloadTraitsT>
class FifoLeaseLifetimeRegistry {
public:
  using StrongLease = StrongLeaseT;
  using PayloadTraits = PayloadTraitsT;
  using Payload = typename PayloadTraits::Payload;

  FifoLeaseLifetimeRegistry() = default;
  FifoLeaseLifetimeRegistry(const FifoLeaseLifetimeRegistry &) = delete;
  FifoLeaseLifetimeRegistry &
  operator=(const FifoLeaseLifetimeRegistry &) = delete;
  FifoLeaseLifetimeRegistry(FifoLeaseLifetimeRegistry &&) noexcept = default;
  FifoLeaseLifetimeRegistry &
  operator=(FifoLeaseLifetimeRegistry &&) noexcept = default;
  ~FifoLeaseLifetimeRegistry() = default;

  /**
   * @brief Push a lease to the back of the FIFO queue.
   * @param lease The lease to add
   */
  void push(StrongLease lease) {
    if (!lease) {
      return;
    }
    leases_.pushBack(std::move(lease));
  }

  /**
   * @brief Release all ready leases from the front of the queue.
   *
   * Scans from the back to find the last ready lease, then releases all
   * leases from head to that point. In debug mode, verifies that all leases
   * in the released range are actually ready (FIFO order validation).
   *
   * @tparam FastOps Operations type providing isCompleted/waitUntilCompleted
   * @return Number of released leases
   */
  template <typename FastOps> std::size_t releaseReady() {
    if (head_ >= leases_.size()) {
      leases_.clear();
      head_ = 0;
      return 0;
    }

    std::size_t ready_end = 0;
    for (std::size_t i = leases_.size(); i > head_; --i) {
      auto &lease = leases_[i - 1];
      auto *payload = lease.operator->();
      if (payload == nullptr) {
        ready_end = i;
        break;
      }
      if (PayloadTraits::template isReady<FastOps>(*payload)) {
        ready_end = i;
        break;
      }
    }

    if (ready_end == 0) {
      return 0;
    }

    const std::size_t released = ready_end - head_;
    for (std::size_t i = head_; i < ready_end; ++i) {
      auto *payload = leases_[i].operator->();
      if (payload != nullptr) {
        PayloadTraits::template validateBeforeRelease<FastOps>(*payload);
      }
      leases_[i].release();
    }
    head_ = ready_end;
    compactIfNeeded();
    return released;
  }

  /**
   * @brief Clear all leases. All must be ready.
   * @tparam FastOps Operations type providing isCompleted
   * @throws OrteafError if any lease is not ready
   */
  template <typename FastOps> void clear() {
    if (head_ >= leases_.size()) {
      leases_.clear();
      head_ = 0;
      return;
    }

    for (std::size_t i = head_; i < leases_.size(); ++i) {
      auto *payload = leases_[i].operator->();
      if (payload == nullptr) {
        continue;
      }
      if (!PayloadTraits::template isReady<FastOps>(*payload)) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "FIFO lease lifetime registry clear aborted due to active leases");
      }
    }

    for (std::size_t i = head_; i < leases_.size(); ++i) {
      leases_[i].release();
    }
    leases_.clear();
    head_ = 0;
  }

  /**
   * @brief Wait for all leases to complete and release them.
   * @tparam FastOps Operations type providing waitUntilCompleted
   * @return Number of released leases
   */
  template <typename FastOps> std::size_t waitUntilReady() {
    if (head_ >= leases_.size()) {
      leases_.clear();
      head_ = 0;
      return 0;
    }

    for (std::size_t i = head_; i < leases_.size(); ++i) {
      auto *payload = leases_[i].operator->();
      if (payload == nullptr) {
        continue;
      }
      PayloadTraits::template waitUntilReady<FastOps>(*payload);
      if (!PayloadTraits::template isReady<FastOps>(*payload)) {
        ::orteaf::internal::diagnostics::error::throwError(
            ::orteaf::internal::diagnostics::error::OrteafErrc::InvalidState,
            "FIFO lease lifetime registry wait failed to complete lease");
      }
    }

    const std::size_t released = leases_.size() - head_;
    for (std::size_t i = head_; i < leases_.size(); ++i) {
      leases_[i].release();
    }
    leases_.clear();
    head_ = 0;
    return released;
  }

  std::size_t size() const noexcept {
    return (head_ >= leases_.size()) ? 0 : (leases_.size() - head_);
  }

  bool empty() const noexcept { return size() == 0; }

#if ORTEAF_ENABLE_TEST
  std::size_t storageSizeForTest() const noexcept { return leases_.size(); }
  std::size_t headIndexForTest() const noexcept { return head_; }
#endif

private:
  void compactIfNeeded() {
    if (head_ == 0) {
      return;
    }
    if (head_ >= leases_.size()) {
      leases_.clear();
      head_ = 0;
      return;
    }
    if (head_ < (leases_.size() / 2)) {
      return;
    }

    const std::size_t new_size = leases_.size() - head_;
    for (std::size_t i = 0; i < new_size; ++i) {
      leases_[i] = std::move(leases_[head_ + i]);
    }
    leases_.resize(new_size);
    head_ = 0;
  }

  ::orteaf::internal::base::HeapVector<StrongLease> leases_{};
  std::size_t head_{0};
};

} // namespace orteaf::internal::base::manager
