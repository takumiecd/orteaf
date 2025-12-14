#pragma once

#include <atomic>
#include <cstdint>

#include <orteaf/internal/runtime/base/lease/category.h>
#include <orteaf/internal/runtime/base/lease/concepts.h>

namespace orteaf::internal::runtime::base {

/// @brief Weak-unique control block - single ownership with weak reference
/// support
/// @details Allows weak references to observe the resource without owning it.
/// The resource is destroyed when the strong owner releases, but control block
/// persists until all weak references are gone.
template <typename PayloadT>
  requires PayloadConcept<PayloadT>
struct WeakUniqueControlBlock {
  using Category = lease_category::WeakUnique;
  using Payload = PayloadT;

  std::atomic<bool> in_use{false};
  std::atomic<std::uint32_t> weak_count{0};
  PayloadT payload{};

  /// @brief Try to acquire exclusive (strong) ownership
  /// @return true if successfully acquired
  bool tryAcquire() noexcept {
    bool expected = false;
    return in_use.compare_exchange_strong(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed);
  }

  /// @brief Release strong ownership
  void release() noexcept { in_use.store(false, std::memory_order_release); }

  /// @brief Acquire a weak reference
  void acquireWeak() noexcept {
    weak_count.fetch_add(1, std::memory_order_relaxed);
  }

  /// @brief Release a weak reference
  /// @return true if this was the last weak reference and resource is not in
  /// use
  bool releaseWeak() noexcept {
    const auto prev = weak_count.fetch_sub(1, std::memory_order_acq_rel);
    return prev == 1 && !in_use.load(std::memory_order_acquire);
  }

  /// @brief Check if strong owner exists
  bool isAlive() const noexcept {
    return in_use.load(std::memory_order_acquire);
  }

  /// @brief Try to promote weak reference to strong
  /// @return true if successfully promoted (same as tryAcquire)
  bool tryPromote() noexcept { return tryAcquire(); }
};

// Verify concept satisfaction
static_assert(ControlBlockConcept<WeakUniqueControlBlock<int>>);
static_assert(WeakableControlBlockConcept<WeakUniqueControlBlock<int>>);
static_assert(PromotableControlBlockConcept<WeakUniqueControlBlock<int>>);

} // namespace orteaf::internal::runtime::base
