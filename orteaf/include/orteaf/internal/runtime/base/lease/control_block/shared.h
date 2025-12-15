#pragma once

#include <atomic>
#include <cstdint>

#include <orteaf/internal/runtime/base/lease/category.h>
#include <orteaf/internal/runtime/base/lease/concepts.h>

namespace orteaf::internal::runtime::base {

/// @brief Shared control block - shared ownership with reference counting
/// @details Multiple leases can share this resource. Uses atomic reference
/// count for thread-safe sharing.
template <typename SlotT>
  requires SlotConcept<SlotT>
struct SharedControlBlock {
  using Category = lease_category::Shared;
  using Slot = SlotT;

  std::atomic<std::uint32_t> strong_count{0};
  SlotT slot{};

  SharedControlBlock() = default;
  SharedControlBlock(const SharedControlBlock &) = delete;
  SharedControlBlock &operator=(const SharedControlBlock &) = delete;

  SharedControlBlock(SharedControlBlock &&other) noexcept
      : slot(std::move(other.slot)) {
    strong_count.store(other.strong_count.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  }

  SharedControlBlock &operator=(SharedControlBlock &&other) noexcept {
    if (this != &other) {
      strong_count.store(other.strong_count.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      slot = std::move(other.slot);
    }
    return *this;
  }

  /// @brief Try to acquire (first acquisition)
  /// @return true if this is the first acquisition (count goes 0->1)
  bool tryAcquire() noexcept {
    std::uint32_t expected = 0;
    return strong_count.compare_exchange_strong(
        expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
  }

  /// @brief Acquire a shared reference (increment count)
  /// @return always true for shared resources
  bool acquire() noexcept {
    strong_count.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  /// @brief Release a shared reference and prepare for reuse if last
  /// @return true if this was the last reference (count goes 1->0)
  /// @note Automatically increments generation if this was the last reference
  bool release() noexcept {
    if (strong_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // This was the last reference, increment generation for reuse
      if constexpr (SlotT::has_generation) {
        slot.incrementGeneration();
      }
      return true;
    }
    return false;
  }

  /// @brief Get current reference count
  std::uint32_t count() const noexcept {
    return strong_count.load(std::memory_order_acquire);
  }

  /// @brief Check if any references exist
  bool isAlive() const noexcept { return count() > 0; }

  /// @brief Check if fully released (count == 0)
  bool isReleased() const noexcept { return count() == 0; }

  /// @brief Mark slot as initialized/valid
  void validate() noexcept {
    if constexpr (SlotT::has_initialized) {
      slot.markInitialized();
    }
  }

  /// @brief Mark slot as uninitialized/invalid
  void invalidate() noexcept {
    if constexpr (SlotT::has_initialized) {
      slot.markUninitialized();
    }
  }

  /// @brief Prepare for reuse - validates state and increments generation
  /// @return true if successfully prepared (was released), false if still in
  /// use
  bool prepareForReuse() noexcept {
    if (!isReleased()) {
      return false;
    }
    if constexpr (SlotT::has_generation) {
      slot.incrementGeneration();
    }
    return true;
  }
};

// Verify concept satisfaction
static_assert(ControlBlockConcept<SharedControlBlock<int>>);
static_assert(SharedControlBlockConcept<SharedControlBlock<int>>);

} // namespace orteaf::internal::runtime::base
