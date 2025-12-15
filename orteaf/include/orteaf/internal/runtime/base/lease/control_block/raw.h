#pragma once

#include <orteaf/internal/runtime/base/lease/category.h>
#include <orteaf/internal/runtime/base/lease/concepts.h>
#include <orteaf/internal/runtime/base/lease/slot.h>

namespace orteaf::internal::runtime::base {

/// @brief Raw control block - no reference counting
/// @details Used for resources that don't need lifecycle management.
/// Initialization state is tracked by the ControlBlock itself.
template <typename SlotT>
  requires SlotConcept<SlotT>
class RawControlBlock {
public:
  using Category = lease_category::Raw;
  using Slot = SlotT;
  using Payload = typename SlotT::Payload;

  RawControlBlock() = default;
  RawControlBlock(const RawControlBlock &) = default;
  RawControlBlock &operator=(const RawControlBlock &) = default;
  RawControlBlock(RawControlBlock &&) = default;
  RawControlBlock &operator=(RawControlBlock &&) = default;

  // =========================================================================
  // Lifecycle API
  // =========================================================================

  /// @brief Always succeeds (no ref counting)
  constexpr bool acquire() noexcept { return true; }

  /// @brief Release and prepare for reuse
  /// @return always true for raw resources
  bool release() noexcept {
    if constexpr (SlotT::has_generation) {
      slot_.incrementGeneration();
    }
    return true;
  }

  /// @brief Always alive (no ref counting)
  constexpr bool isAlive() const noexcept { return true; }

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
  SlotT slot_{};
};

// Verify concept satisfaction
static_assert(ControlBlockConcept<RawControlBlock<RawSlot<int>>>);

} // namespace orteaf::internal::runtime::base
