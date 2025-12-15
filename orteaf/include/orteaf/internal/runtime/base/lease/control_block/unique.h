#pragma once

#include <atomic>

#include <orteaf/internal/runtime/base/lease/category.h>
#include <orteaf/internal/runtime/base/lease/concepts.h>
#include <orteaf/internal/runtime/base/lease/slot.h>

namespace orteaf::internal::runtime::base {

/// @brief Unique control block - single ownership with in_use flag
/// @details Only one lease can hold this resource at a time.
/// Uses atomic CAS for thread-safe acquisition.
/// Initialization state is tracked by the ControlBlock itself.
template <typename SlotT>
  requires SlotConcept<SlotT>
class UniqueControlBlock {
public:
  using Category = lease_category::Unique;
  using Slot = SlotT;
  using Payload = typename SlotT::Payload;

  UniqueControlBlock() = default;
  UniqueControlBlock(const UniqueControlBlock &) = delete;
  UniqueControlBlock &operator=(const UniqueControlBlock &) = delete;

  UniqueControlBlock(UniqueControlBlock &&other) noexcept
      : initialized_(other.initialized_), slot_(std::move(other.slot_)) {
    in_use_.store(other.in_use_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  }

  UniqueControlBlock &operator=(UniqueControlBlock &&other) noexcept {
    if (this != &other) {
      initialized_ = other.initialized_;
      in_use_.store(other.in_use_.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
      slot_ = std::move(other.slot_);
    }
    return *this;
  }

  // =========================================================================
  // Lifecycle API
  // =========================================================================

  /// @brief Acquire exclusive ownership
  /// @return true if successfully acquired, false if already in use
  bool acquire() noexcept {
    bool expected = false;
    return in_use_.compare_exchange_strong(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed);
  }

  /// @brief Release ownership
  /// @return true if was in use and now released, false if wasn't in use
  /// @note Automatically increments generation if supported
  bool release() noexcept {
    bool expected = true;
    if (in_use_.compare_exchange_strong(expected, false,
                                        std::memory_order_release,
                                        std::memory_order_relaxed)) {
      if constexpr (SlotT::has_generation) {
        slot_.incrementGeneration();
      }
      return true;
    }
    return false;
  }

  /// @brief Check if currently in use
  bool isAlive() const noexcept {
    return in_use_.load(std::memory_order_acquire);
  }

  // =========================================================================
  // Initialization State (managed by ControlBlock)
  // =========================================================================

  /// @brief Check if resource is initialized
  bool isInitialized() const noexcept { return initialized_; }

  /// @brief Mark resource as initialized/valid
  void validate() noexcept { initialized_ = true; }

  /// @brief Mark resource as uninitialized/invalid
  void invalidate() noexcept { initialized_ = false; }

  // =========================================================================
  // Payload Access
  // =========================================================================

  /// @brief Access the payload
  Payload &payload() noexcept { return slot_.get(); }
  const Payload &payload() const noexcept { return slot_.get(); }

  // =========================================================================
  // Generation (delegated to Slot)
  // =========================================================================

  /// @brief Get current generation (0 if not supported)
  auto generation() const noexcept { return slot_.generation(); }

private:
  bool initialized_{false};
  std::atomic<bool> in_use_{false};
  SlotT slot_{};
};

// Verify concept satisfaction
static_assert(ControlBlockConcept<UniqueControlBlock<RawSlot<int>>>);

} // namespace orteaf::internal::runtime::base
